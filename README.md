
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
