import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.dataset_heart import Heart_dataset, RandomGenerator
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vit_seg_modeling import VisionTransformer as trasUnet
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
from config import get_config
from utils import calculate_metric
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import copy
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
            image = cv2.resize(image, (self.output_size[0], self.output_size[1]), interpolation=cv2.INTER_CUBIC)
#             image = zoom(image, (self.height / x, self.width / y, 1), order=3)  # why not 3?
        return image
    
    def apply_to_mask(self, mask, **params):
        x, y = mask.shape
        if x != self.height or y != self.width:
            mask = zoom(mask, (self.height / x, self.width / y), order=0)
        return mask

    def get_transform_init_args_names(self):
        return ("height", "width")

parser = argparse.ArgumentParser()
parser.add_argument('--A2C_path', type=str,
                    default='../data/validation/A2C', help='validation A2C dir')
parser.add_argument('--A4C_path', type=str,
                    default='../data/validation/A4C', help='validation A4C dir')
parser.add_argument('--model', type=str,
                    default='swin_unet', help='model')
parser.add_argument('--dataset', type=str,
                    default='Heart', help='experiment_name')
parser.add_argument('--ckpt_path_A2C', type=str, help='checkpoint path for A2C model')
parser.add_argument('--ckpt_path_A4C', type=str, help='checkpoint path for A4C model')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='model_out', help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_save', default=False, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', 
                    metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()
config = get_config(args)


def inference(args, model, dataset, data_name=None):
   
    testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("The length of train set is: {}".format(len(dataset)))
    logging.info("{} {} test iterations per epoch".format(data_name, len(testloader)))
    
    model.eval()
    total_metric = 0.0
    total_prob = []
    total_label = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
            
            outputs = model(image_batch)
            prob_out = torch.softmax(outputs, dim=1)
            total_prob.append(prob_out.cpu().detach().numpy())
            
            out = torch.argmax(outputs, dim=1)
            prediction = out.cpu().detach().numpy()
            label = label_batch.cpu().detach().numpy()
            total_label.append(label)
            
            metric_list = []
            for i in range(1, args.num_classes):
                dice, ji, dice2, ji2 = calculate_metric(prediction == i, label == i)
                metric_list.append(np.array([dice, ji, dice2, ji2]))
            total_metric += np.array(metric_list)
            
        
        avg_metric = np.array(total_metric) / len(dataset)
        
        if args.num_classes > 2:
            mean_dice = np.mean(avg_metric, axis=0)[0]
            mean_JI = np.mean(avg_metric, axis=0)[1]
            logging.info('Testing performance : mean_dice : %f mean_JI : %f' % (mean_dice, mean_JI))
    
#     print(avg_metric)
    total_prob = np.concatenate(total_prob, 0)
    total_label = np.concatenate(total_label, 0)
    return total_prob, total_label


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Heart': {
            'Dataset': Heart_dataset,
            'root_path': args.A2C_path,
            'num_classes': 2,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.is_pretrain = True

    if args.model == 'swin_unet':
        net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    elif args.model == 'trans_unet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
            print(config_vit.patches.size, config_vit.patches.grid)
        net = trasUnet(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
    
    snapshot = args.ckpt_path_A2C
    test_num = snapshot.split('/')[-2]
    snapshot_name = snapshot.split('/')[-1]
    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+test_num+'_'+snapshot_name+".txt", level=logging.INFO, 
                        filemode = "w", format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    
    if args.is_save:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    
    base = RandomGenerator(output_size=[args.img_size, args.img_size], train=False)
    transform_list = [ ['o', 'Hflip', 'Vflip', 'angle_2'], ['o', 'Hflip', 'Vflip', 'angle_2'] ]
    data_path_list = [args.A2C_path, args.A4C_path]
    model_ckpt_list = [args.ckpt_path_A2C, args.ckpt_path_A4C]
    
    
    to_tensor = ToTensorV2()


    test_transform = [
                to_tensor
            ]
    scale = [A.Resize(args.img_size, args.img_size, interpolation=2)]
    
    model_ensemble = [0.0, 0.0] # 누적 probs
    ensemble_score = [0.0, 0.0]
    o_label_list = [0, 0] # d1_label, d2_label
    
    final_score = [0.0, 0.0]
    for i in range(len(data_path_list)):
        data_path = data_path_list[i]
        print('========DATA', data_path.split('/')[-1])
        snapshot = model_ckpt_list[i]
        msg = net.load_state_dict(torch.load(snapshot))
        print("self trained net", msg)
        test_num = snapshot.split('/')[-2]
        snapshot_name = snapshot.split('/')[-1]
        print(test_num, snapshot_name)
        
        tta_probs = 0.0 # 누적 probs
        o_label = 0
        
        for t_idx, t in enumerate(transform_list[i]):
            print('transforms', t)
            tt = t.split('_')[0]
            if tt == 'o':
                test_T = transforms.Compose([base])
#                 test_T = A.Compose(scale + test_transform)
            elif tt == 'Hflip':
                cb = copy.deepcopy(base)
                cb.Hflip = True
                test_T = transforms.Compose([cb])
            elif tt == 'Vflip':
                cb = copy.deepcopy(base)
                cb.Vflip = True
                test_T = transforms.Compose([cb])
            elif tt == 'rot90':
                k = int(t.split('_')[1])
                cb = copy.deepcopy(base)
                cb.rot90 = k
                test_T = transforms.Compose([cb])
            elif tt == 'angle':
                angle = int(t.split('_')[1])
                cb = copy.deepcopy(base)
                cb.angle = angle
                test_T = transforms.Compose([cb])
            print(test_T)
            dataset = args.Dataset(base_dir=data_path, transform=test_T)
            probs, label = inference(args, net, dataset, data_name=data_path.split('/')[-1])
            pred = np.argmax(probs, 1)
            dice, ji, dice2, ji2 = calculate_metric(pred == 1, label == 1)
#             print(dice, ji, dice2, ji2)
            logging.info('T : %s mean_dice %f mean_JI %f' % (t, dice, ji))
            # probs (100,2,512,512)
            if tt == 'o':
                o_label = label
                o_label_list[i] = label
            elif tt == 'Hflip':
                probs = np.flip(probs, axis=3)
            elif tt == 'Vflip':
                probs = np.flip(probs, axis=2)
            elif tt == 'rot90':
                k = int(t.split('_')[1])
                probs = np.rot90(probs, -k, axes=(2, 3))
            elif tt == 'angle':
                angle = int(t.split('_')[1])
                probs = ndimage.rotate(probs, -angle, axes=(3, 2), order=0, reshape=False)
            tta_probs += probs
            
            if t_idx > 0:
                tta_pred = np.argmax(tta_probs / (t_idx+1), 1)
                dice, ji, dice2, ji2 = calculate_metric(tta_pred == 1, o_label == 1)
                logging.info('TTA : %s mean_dice %f mean_JI %f' % (transform_list[i][:t_idx+1], dice, ji))
                if t_idx == len(transform_list)-1:
                    final_score[0] += dice
                    final_score[1] += ji
        
        model_ensemble[i] = (tta_probs / len(transform_list[i]))
    logging.info('Final mean_dice %f mean_JI %f' % (final_score[0] / 2, final_score[1] / 2))
    
    args.img_size = 512
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        print(config_vit.patches.size, config_vit.patches.grid)
    net = trasUnet(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
    base = RandomGenerator(output_size=[args.img_size, args.img_size], train=False)
    
    print('\nModel2!!')
    transform_list = [ ['o', 'Hflip', 'angle_1'], ['o', 'Hflip'] ]
    model_ckpt_list = ['./transunet_model_out/test19/epoch_299.pth', './transunet_model_out/test19/epoch_299.pth']
    final_score = [0.0, 0.0]
    for i in range(len(data_path_list)):
        data_path = data_path_list[i]
        print('========DATA', data_path.split('/')[-1])
        snapshot = model_ckpt_list[i]
        msg = net.load_state_dict(torch.load(snapshot))
        print("self trained net", msg)
        test_num = snapshot.split('/')[-2]
        snapshot_name = snapshot.split('/')[-1]
        print(test_num, snapshot_name)
        
        tta_probs = 0.0 # 누적 probs
        o_label = 0
        
        for t_idx, t in enumerate(transform_list[i]):
            print('transforms', t)
            tt = t.split('_')[0]
            if tt == 'o':
                test_T = transforms.Compose([base])
#                 test_T = A.Compose(scale + test_transform)
            elif tt == 'Hflip':
                cb = copy.deepcopy(base)
                cb.Hflip = True
                test_T = transforms.Compose([cb])
            elif tt == 'Vflip':
                cb = copy.deepcopy(base)
                cb.Vflip = True
                test_T = transforms.Compose([cb])
            elif tt == 'rot90':
                k = int(t.split('_')[1])
                cb = copy.deepcopy(base)
                cb.rot90 = k
                test_T = transforms.Compose([cb])
            elif tt == 'angle':
                angle = int(t.split('_')[1])
                cb = copy.deepcopy(base)
                cb.angle = angle
                test_T = transforms.Compose([cb])
            print(test_T)
            dataset = args.Dataset(base_dir=data_path, transform=test_T)
            probs, label = inference(args, net, dataset, data_name=data_path.split('/')[-1])
            pred = np.argmax(probs, 1)
            dice, ji, dice2, ji2 = calculate_metric(pred == 1, label == 1)
#             print(dice, ji, dice2, ji2)
            logging.info('T : %s mean_dice %f mean_JI %f' % (t, dice, ji))
            # probs (100,2,512,512)
            if tt == 'o':
                o_label = label
            elif tt == 'Hflip':
                probs = np.flip(probs, axis=3)
            elif tt == 'Vflip':
                probs = np.flip(probs, axis=2)
            elif tt == 'rot90':
                k = int(t.split('_')[1])
                probs = np.rot90(probs, -k, axes=(2, 3))
            elif tt == 'angle':
                angle = int(t.split('_')[1])
                probs = ndimage.rotate(probs, -angle, axes=(3, 2), order=0, reshape=False)
            
#             probs = zoom(probs, (1, 1,512 / args.img_size, 512 / args.img_size), order=0) # 384 to 512
            tta_probs += probs
            
            if t_idx > 0:
                tta_pred = np.argmax(tta_probs / (t_idx+1), 1)
                dice, ji, dice2, ji2 = calculate_metric(tta_pred == 1, o_label_list[i] == 1)
                logging.info('TTA : %s mean_dice %f mean_JI %f' % (transform_list[i][:t_idx+1], dice, ji))
                if t_idx == len(transform_list)-1:
                    final_score[0] += dice
                    final_score[1] += ji
                    
        model_ensemble[i] += (tta_probs / len(transform_list[i]))
        model_pred = np.argmax(model_ensemble[i] / 2, 1)
        dice, ji, dice2, ji2 = calculate_metric(model_pred == 1, o_label_list[i] == 1)
        ensemble_score[0] += dice
        ensemble_score[1] += ji
        logging.info('%s Model E : mean_dice %f mean_JI %f' % (data_path.split('/')[-1], dice, ji))
        
    logging.info('Final mean_dice %f mean_JI %f' % (ensemble_score[0] / 2, ensemble_score[1] / 2))

