#buffer for FTD 
CUDA_VISIBLE_DEVICES=5 python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=10 --zca \
--buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv" \
--rho_max=0.01 --rho_min=0.01 

cd /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main/buffer 
conda activate ztl
# watch -n 1 nvidia-smi  
# python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=10 --zca --buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv"


CUDA_VISIBLE_DEVICES=5 python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=10 --zca \
--buffer_path="/home/ssd7T/ztl_dm/mtt" --data_path="/home/ssd7T/ZTL_gcond/data_cv" \
--rho_max=0.01 --rho_min=0.01 