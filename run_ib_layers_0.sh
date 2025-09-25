# federated training and attack measurement calculatinging command 
dataset=cifar10 # or [cifar100, dermnet, cifar10]
# model_name=ResNet18_IB_layer # or [ResNet18, ResNet18_IB, ResNet18_IB_Block, ResNet18_IB_layer, alexnet]
model_name=VGG16
layer=0

opt=sgd
seed=1 
lr=0.001
local_epoch=1
dynamic_ib=ir
ib_beta=1e-3

# ib_beta=0.00001


# iid experiment
save_dir=log_fedmia/iid_ib_MI
# CUDA_VISIBLE_DEVICES=1

for ib_beta in 1e-3
do
    python main_test_global_MI.py --seed $seed --num_users 5 --iid 1 --ib_costum $ib_beta --ib_beta $ib_beta --ib_model_layer $layer \
        --dataset $dataset --model_name $model_name --epochs 100 --local_ep $local_epoch \
        --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
        --lr_up cosine --MIA_mode 0  --gpu 1
    # ./upload_to_onedrive.sh ./log_fedmia exp/
done
# -m debugpy --listen 5010 --wait-for-client

# non-iid experiment
# save_dir=log_fedmia/noniid_ib_MI
# # CUDA_VISIBLE_DEVICES=0

# for dynamic_ib in ir 
# do
#     for bt in 10
#     do
#         echo "Running with noniid beta $bt, IB beta $ib_beta"
#         python  main_test_global_MI.py --seed $seed --num_users 10 --iid 0 --beta $bt --ib_costum $ib_beta --ib_beta $ib_beta --ib_model_layer $layer --dynamic_ib $dynamic_ib\
#         --dataset $dataset --model_name $model_name --epochs 100 --local_ep 4 \
#         --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
#         --lr_up cosine --MIA_mode 0  --gpu 1

#         # ./upload_to_onedrive.sh ./log_fedmia exp/
#     done
# done

# -m debugpy --listen 5010 --wait-for-client
# --ib_costum $ib_beta --ib_beta $ib_beta

# python -m debugpy --listen 5010 --wait-for-client main.py --seed $seed --num_users 10 --iid 0 --beta 10 --ib_costum $ib_beta --ib_beta $ib_beta\
#     --dataset $dataset --model_name $model_name --epochs 300 --local_ep $local_epoch \
#     --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
#     --lr_up cosine --MIA_mode 1  --gpu 0