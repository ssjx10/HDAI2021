# Heart Disease AI Datathon 2021

Project 기간: 2021년 11월 26일 → 2021년 12월 7일

[HDAI_DATATHON](http://hdaidatathon.com/)

# Overview

- “2021 인공지능 학습용 데이터 구축사업”의 일환으로 추진된 인공지능 학습용 심장질환 심초음파 및 심전도 데이터셋을 이용하여 심초음파/심전도 질환을 판별하는 AI 진단 모델링 경진대회
- 주제 1 - 심초음파 데이터셋을 활용한 **좌심실 분할 AI모델**
    - Apical 2 chamber(A2C) & Apical 4 chamber(A4C) view 이미지를 활용해 좌심실 분할하는 딥러닝 모델 개발

## Train
model - 'swin_unet' / 'trans_unet'
```bash
python train.py --model 'trans_unet' --vit_name [VIT_NAME] --root_path [DATA_DIR] --valid_path [VALID_DIR] --max_epochs [EPOCH_TIME] --output_dir [OUT_DIR] --img_Hsize [img_Hsize] --img_Wsize [img_Wsize]  --base_lr [LEARNING_RATE] --batch_size [BATCH_SIZE] --n_gpu [N_GPU] --angle [ANGLE]
```

## Test 
```bash
python test.py --A2C_path [A2C_path] --A4C_path [A4C_path] --model 'trans_unet' --ckpt_path_m1 [M1_path] --ckpt_path_m2 [M2_path] --img_Hsize 512 --img_Wsize 512 --batch_size 64
```

## Infer 
```bash
python infer.py --model 'trans_unet' --ckpt_path_m1 [M1_path] --ckpt_path_m2 [M2_path] --img_Hsize 512 --img_Wsize 512 --batch_size 32
```
A2C_path와 A4C_path 수정!!

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
