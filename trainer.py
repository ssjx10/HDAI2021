import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume, calculate_metric_percase, calculate_metric

from scipy.ndimage.interpolation import zoom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
import cv2

class Zoom(DualTransform):
    def __init__(self, height, width, always_apply=False, p=1):
        super(Zoom, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.always_apply = always_apply
        self.p = p
        
    def apply(self, image, **params):
        x, y, _ = image.shape
        if x != self.height or y != self.width:
            image = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
#             image = zoom(image, (self.height / x, self.width / y, 1), order=3)  # why not 3?
        return image
    
    def apply_to_mask(self, mask, **params):
        x, y = mask.shape
        if x != self.height or y != self.width:
            mask = zoom(mask, (self.height / x, self.width / y), order=0)
        return mask

    def get_transform_init_args_names(self):
        return ("height", "width")

H_flip = A.HorizontalFlip(p=0.5)
V_flip = A.VerticalFlip(p=0.5)
rot90 = A.RandomRotate90(p=0.5)
rotate = A.Rotate(limit=10, interpolation=0)
cutout = A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5)
blur = A.Blur(blur_limit=24, p=0.5)
ssr = A.ShiftScaleRotate(rotate_limit=15, p=0.5)
# normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
to_tensor = ToTensorV2()


train_transform = [
            H_flip,
            V_flip,
            rot90,
            ssr,
        ]

test_transform = [
            to_tensor
        ]

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_heart import Heart_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, filemode = "w",
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    train_T = transforms.Compose([RandomGenerator(output_size=args.img_size, angle=args.angle)])
    test_T = transforms.Compose([RandomGenerator(output_size=args.img_size, rot90=None, angle=None, train=False)])
    
#     scale = [Zoom(args.img_size, args.img_size)]
#     train_T = A.Compose(train_transform + scale + [to_tensor])
#     test_T = A.Compose(scale + test_transform)
    logging.info('train_T {}'.format(str(train_T)))
    logging.info('test_T {}'.format(str(test_T)))
    
    db_train = Heart_dataset(base_dir=args.root_path, transform=train_T)
    db_valid = Heart_dataset(base_dir=args.valid_path, transform=test_T)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            
            if iter_num % 100 == 0:
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                writer.add_scalar('info/loss_dice', loss_dice, iter_num)

                logging.info('----iteration %d - lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' 
                             % (iter_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 100 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        # valid
        if (epoch_num + 1) % args.eval_period == 0:
            model.eval()
            total_metric = 0.0
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(validloader):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    
                    out = torch.argmax(outputs, dim=1)
                    prediction = out.cpu().detach().numpy()
                    label = label_batch.cpu().detach().numpy()
                    
                    metric_list = []
                    for i in range(1, args.num_classes):
                        dice, ji, dice2, ji2 = calculate_metric(prediction == i, label == i)
                        metric_list.append(np.array([dice, ji, dice2, ji2]))
                    total_metric += np.array(metric_list)
            
            avg_metric = np.array(total_metric) / len(db_valid)
            logging.info(' Eval - loss_ce: %f, loss_dice: %f, Dice: %.4f, JI: %.4f' 
                             % (loss_ce.item(), loss_dice.item(), avg_metric[0][0], avg_metric[0][1]))
            
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % args.check_period == 0:
            if args.n_gpu > 1:
                m = model.module
            else:
                m = model
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(m.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            if args.n_gpu > 1:
                m = model.module
            else:
                m = model
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(m.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"