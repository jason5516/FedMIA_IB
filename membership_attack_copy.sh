# membership inference attack command
# path="log_fedmia/noniid"
path="log_fedmia/iid_ib"
seed=2025
total_epoch=300
gpu=0
atk=0
python -m debugpy --listen 5010 --wait-for-client mia_attack_auto_copy.py  ${path} ${total_epoch} ${gpu} ${seed} ${atk} 
# python -u mia_attack_auto_copy.py  ${path} ${total_epoch} ${gpu} ${seed} ${atk}
