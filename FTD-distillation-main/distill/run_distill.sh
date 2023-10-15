#!/bin/bash
# watch -n 1 nvidia-smi  

CUDA_VISIBLE_DEVICES="4,5" python /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main/distill/distill_FTD.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv"  --ema_decay=0.9995 --Iteration=2000 --begain=0

CUDA_VISIBLE_DEVICES="6,7" python /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main/distill/distill_FTD.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv"  --ema_decay=0.9995 --Iteration=2000 --begain=50



CUDA_VISIBLE_DEVICES="6,7" python /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main/distill/distill_FTD.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv"  --ema_decay=0.9995 --Iteration=2000 --begain=50

CUDA_VISIBLE_DEVICES="7,1" python /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main/distill/distill_FTD.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv"  --ema_decay=0.9995 --Iteration=2000 --begain=75




CUDA_VISIBLE_DEVICES=4 python /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main/distill/distill_FTD.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv"  --ema_decay=0.9995 --Iteration=2000 --begain=0






CUDA_VISIBLE_DEVICES=0 python distill_FTD.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --zca \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer_storage/ --data_path=../dataset/ --ema_decay=0.9995 --Iteration=5000

