# federated training and attack measurement calculatinging command 
dataset=cifar100 # or [cifar100, dermnet, cifar10]
model_name=ResNet18_IB_Block # or [ResNet18, ResNet18_IB, ResNet18_IB_Block, alexnet]

opt=sgd
seed=1 
lr=0.001
local_epoch=1

ib_beta=1e-6


# iid experiment
save_dir=log_fedmia/iid_ib
# CUDA_VISIBLE_DEVICES=1
for ib_beta in 1e-6 1e-7 1e-8
do
    python main.py --seed $seed --num_users 10 --iid 1 --ib_costum $ib_beta --ib_beta $ib_beta \
        --dataset $dataset --model_name $model_name --epochs 300 --local_ep $local_epoch \
        --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
        --lr_up cosine --MIA_mode 1  --gpu 1
    
    # ./upload_to_onedrive.sh ./log_fedmia exp/
    # rm -r ./log_fedmia
done

# non-iid experiment
# save_dir=log_fedmia/noniid_ib/noniid
# CUDA_VISIBLE_DEVICES=0


# for bt in 0.1 1 10
# do
#     echo "Running with noniid beta $bt, IB beta $ib_beta"
#     python  main.py --seed $seed --num_users 10 --iid 0 --beta $bt --ib_costum $ib_beta --ib_beta $ib_beta\
#     --dataset $dataset --model_name $model_name --epochs 100 --local_ep $local_epoch \
#     --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
#     --lr_up cosine --MIA_mode 1  --gpu 0
# done

# -m debugpy --listen 5010 --wait-for-client
# --ib_costum $ib_beta --ib_beta $ib_beta

# python -m debugpy --listen 5010 --wait-for-client main.py --seed $seed --num_users 10 --iid 0 --beta 10 --ib_costum $ib_beta --ib_beta $ib_beta\
#     --dataset $dataset --model_name $model_name --epochs 300 --local_ep $local_epoch \
#     --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
#     --lr_up cosine --MIA_mode 1  --gpu 0