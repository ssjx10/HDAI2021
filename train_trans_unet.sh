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
    OUT_DIR='./transunet_model_out/test27'
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
    DATA_DIR='../data/train/A2C_A4C'
fi

if [ $learning_rate ]
then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.01
fi

if [ $batch_size ]
then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=12
fi

echo "start train model"

python train.py --model 'trans_unet' --vit_name $VIT_NAME --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_Hsize 512 --img_Wsize 512 --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --n_gpu 2 --angle 5
