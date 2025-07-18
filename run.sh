# federated training and attack measurement calculatinging command 
dataset=cifar100 # or [cifar100, dermnet, cifar10]
model_name=ResNet18 # or [ResNet18, ResNet18_IB, alexnet]

opt=sgd
seed=1 
lr=0.001
local_epoch=4
bt=1.0

# iid experiment
# save_dir=log_fedmia/iid
# CUDA_VISIBLE_DEVICES=1

# python main.py --seed $seed --num_users 10 --iid 1 --defense p2protect\
#     --dataset $dataset --model_name $model_name --epochs 100 --local_ep $local_epoch \
#     --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
#     --lr_up cosine --MIA_mode 1  --gpu 1

# python main.py --seed $seed --num_users 10 --iid 1 --defense none --dp --sigma_sgd 0.3\
#     --dataset $dataset --model_name $model_name --epochs 100 --local_ep $local_epoch \
#     --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
#     --lr_up cosine --MIA_mode 1  --gpu 1

# non-iid experiment
save_dir=log_fedmia/cifar100_noniid
# CUDA_VISIBLE_DEVICES=1

for df in p2protect FedDPA none mix_up instahide
do
    for bt in 1.0 10.0 100.0
    do
        python  main.py --seed $seed --num_users 10 --iid 0 --beta $bt --defense $df\
        --dataset $dataset --model_name $model_name --epochs 40 --local_ep $local_epoch \
        --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
        --lr_up cosine --MIA_mode 1  --gpu 1
    done
done

# -m debugpy --listen 5010 --wait-for-client
# --ib_costum $ib_beta --ib_beta $ib_beta
#  mix_up instahide quant sparse