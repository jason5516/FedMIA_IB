# federated training and attack measurement calculatinging command 
dataset=cifar10 # or [cifar100, dermnet, cifar10]
model_name=ResNet18_IB_layer # or [ResNet18, ResNet18_IB, ResNet18_IB_Block, alexnet]

opt=sgd
seed=1 
lr=1e-3
local_epoch=1
bt=0.1
ib_beta=1e-6
n_class=10
# non-iid experiment
save_dir=log_fedmia/noniid_ib/noniid
CUDA_VISIBLE_DEVICES=1
dynamic_ib=entropy


for n_class in 4
do
    echo "Running with n_class $n_class, IB beta $ib_beta"
    python main.py --seed $seed --num_users 5 --iid 2 --n_classes $n_class --beta $bt --ib_costum $ib_beta --ib_beta $ib_beta ---ib_model_layer 0\
    --dataset $dataset --model_name $model_name --epochs 100 --local_ep $local_epoch \
    --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
    --lr_up cosine --MIA_mode 1  --gpu 1

    # ./upload_to_onedrive.sh ./log_fedmia exp/
    # rm -r ./log_fedmia
done

# -m debugpy --listen 5010 --wait-for-client

