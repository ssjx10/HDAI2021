#!/usr/bin/env bash
if [ ${epoch_time} ]
then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=250
fi

if [ $out_dir ]
then
    OUT_DIR=$out_dir
else
    OUT_DIR='./seper_transunet_model_out/test4_A4C'
fi

if [ $vit_name ]
then
    VIT_NAME=$vit_name
else
    VIT_NAME='R50-ViT-B_16'
fi

if [ $data_dir ]
then
    DATA_DIR=$data_dir
else
    DATA_DIR='../data/train/A4C'
fi

if [ $valid_dir ]
then
    VALID_DIR=$valid_dir
else
    VALID_DIR='../data/validation/A4C'
fi

if [ $learning_rate ]
then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.01
fi

if [ $img_size ]
then
    IMG_SIZE=$img_size
else
    IMG_SIZE=512
fi

if [ $batch_size ]
then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=12
fi

echo "start train model"

CUDA_VISIBLE_DEVICES=1 python train.py --model 'trans_unet' --vit_name $VIT_NAME --root_path $DATA_DIR --valid_path $VALID_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --angle 15
