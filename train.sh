source activate anno
CUDA_VISIBLE_DEVICES=1 \
nohup python train.py --config "./configs/unet_base_20.yaml">unet_base_20.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 \
nohup python train.py --config "./configs/unet_base_leakyrelu20_2.yaml">unet_base_leakyrelu20_2.out 2>&1 &