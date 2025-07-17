# federated training and attack measurement calculatinging command 
dataset=cifar100 # or [cifar100, dermnet, cifar10]
model_name=ResNet18 # or [ResNet18, ResNet18_IB, ResNet18_IB_Block, ResNet18_IB_layer, alexnet]
layer=0

opt=sgd
seed=1 
lr=0.001
local_epoch=1
dynamic_ib=ir
ib_beta=1e-5


# ib_beta=0.00001


# iid experiment
save_dir=log_fedmia/noniid_ib
# CUDA_VISIBLE_DEVICES=1

for bt in 1.0 
do
    python main.py --seed $seed --num_users 10 --iid 0 --beta $bt --defense FedDPA\
        --dataset $dataset --model_name $model_name --epochs 100 --local_ep $local_epoch \
        --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
        --lr_up cosine --MIA_mode 1  --gpu 0
    # ./upload_to_onedrive.sh ./log_fedmia exp/
done