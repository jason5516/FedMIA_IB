import os
from utils.args import parser_args
from utils.datasets import *
import copy
import random
from tqdm import tqdm
import numpy as np
import math
from scipy import spatial
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
import torch.optim as optim
import json

import torch.nn as nn
import torch.nn.functional as F
import models as models

from opacus import PrivacyEngine
from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from experiments.utils import quant
from mia_attack import lira_attack_ldh_cosine, cos_attack
from experiments.defense_p2protect import defend_local_model
from experiments.defense_FedDPA import DefenseStrategy

class FederatedLearning(Experiment):
    """
    Perform federated learning
    """
    def __init__(self, args):
        super().__init__(args) # define many self attributes from args
        self.args = args
        self.watch_train_client_id=0
        self.watch_val_client_id=1

        if "IB" in args.model_name:
            if len(args.ib_costum) < self.num_users:
                args.ib_costum += [args.ib_beta] * (self.num_users - len(args.ib_costum))

            self.user_ib = args.ib_costum

        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3
        self.optim=args.optim
        self.dp = args.dp
        self.defense=args.defense
        self.sigma = args.sigma
        self.cosine_attack =args.cosine_attack  
        self.sigma_sgd = args.sigma_sgd
        self.grad_norm=args.grad_norm
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.data_root = args.data_root
 
        print('==> Preparing data...')
        self.train_set, self.test_set, self.train_set_mia, self.test_set_mia, self.dict_users, self.train_idxs, self.val_idxs, self.disturibute = get_data(dataset=self.dataset,
                                                        data_root = self.data_root,
                                                        iid = self.iid,
                                                        num_users = self.num_users,
                                                        data_aug=self.args.data_augment,
                                                        noniid_beta=self.args.beta,
                                                        save_path = args.save_dir,
                                                        n_class=args.n_classes
                                                        )

        print(len(self.train_set), len(self.test_set))
        print(len(self.train_idxs[0]), len(self.train_idxs[1]))
        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        elif self.args.dataset == 'dermnet':
            self.num_classes = 23
     
        self.MIA_trainset_dir=[]
        self.MIA_valset_dir=[]
        self.MIA_trainset_dir_cos=[]
        self.MIA_valset_dir_cos=[]
        self.train_idxs_cos=[]
        self.testset_idx=(50000+np.arange(10000)).astype(int) # The last 10,000 samples are used as the test set
        # self.testset_idx_cos=(50000+np.arange(1000)).astype(int)

        print('==> Preparing model...')

        self.logs = {'train_acc': [], 'train_sign_acc':[], 'train_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'keys':[],

                     'best_test_acc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }

        self.construct_model()

        if "IB" in args.model_name:
            with open(self.save_dir + '/client_custum_beta.json', 'w') as f:
                json.dump(self.user_ib, f, indent=4)
        
        self.w_t = copy.deepcopy(self.model.state_dict())

        self.trainer = TrainerPrivate(self.model, self.train_set, self.device, self.dp, self.sigma,self.num_classes, self.defense,args.klam,args.up_bound,args.mix_alpha, self.grad_norm, self.sigma_sgd)
        self.tester = TesterPrivate(self.model, self.device)

        if "IB" in self.args.model_name:
            self.trainer.ib = True
        
        if self.defense == 'FedDPA':
            self.defense_strategy = DefenseStrategy(args)

    def compute_C1(self, class_counts):
        class_counts = [c for c in class_counts if c > 0]
        total = sum(class_counts)
        if total == 0 or len(class_counts) <= 1:
            return 1.0
        probs = [c / total for c in class_counts]
        entropy = -sum([p * np.log(p) for p in probs])
        norm_entropy = entropy / np.log(self.num_classes)
        
        return 1 - norm_entropy

    def compute_C2(self, class_counts):
        class_counts = [c for c in class_counts if c > 0]
        n_c = self.num_classes
        n = sum(class_counts)
        if n_c <= 1 or n == 0:
            return 1.0
        ir = 0
        for c_i in class_counts:
            if n - c_i == 0:
                continue
            ir += c_i / (n - c_i)
        ir *= (n_c - 1) / n_c
        if ir <= 1:
            return 0.0
        return 1 - 1 / ir
    
    def get_dynemic_beta(self, c_score):
        ib_max = 1e-2
        ib_min = 1e-5
        scale = ib_max - ib_min

        return [ib_min + (scale * i) for i in c_score] 

    def construct_model(self):

        if self.args.model_name == "ResNet18_IB_layer":
            model = models.__dict__[self.args.model_name](num_classes=self.num_classes, ib_layer_pos=args.ib_model_layer)
        else:
            model = models.__dict__[self.args.model_name](num_classes=self.num_classes)

        
        if self.args.dynamic_ib and self.iid in [0,2,3]:
            c_score = []
            if self.args.dynamic_ib == "entropy":
                for client_id, class_dist in self.disturibute.items():
                    counts = list(class_dist.values())
                    c1_score = self.compute_C1(counts)
                    c_score.append(c1_score) 

            elif self.args.dynamic_ib == "ir":
                for client_id, class_dist in self.disturibute.items():
                    counts = list(class_dist.values())
                    c2_score = self.compute_C2(counts)
                    c_score.append(c2_score) 
                
            
            # 計算costum的IB值
            self.user_ib = self.get_dynemic_beta(c_score)
        
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        
        torch.backends.cudnn.benchmark = True
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    def train(self):
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size , shuffle=False, num_workers=2)
        test_ldr = DataLoader(self.test_set, batch_size=self.batch_size , shuffle=False, num_workers=2)

        # local_train_ldrs = []
        # if args.iid == 1:
        #     for i in range(self.num_users):
        #         if args.defense=='instahide':
        #             self.batch_size=len(self.dict_users[i])
        #             # batch_size=len(self.dict_users[i])
        #             # print("batch_size:",self.batch_size) 5000
        #         local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]), batch_size = self.batch_size,
        #                                         shuffle=True, num_workers=2)
        #         # print("len:",len(local_train_ldr)) 1
        #         local_train_ldrs.append(local_train_ldr)

        # else: 
        #     for i in range(self.num_users):
        #         local_train_ldr = DataLoader(self.dict_users[i], batch_size = self.batch_size,
        #                                         shuffle=True, num_workers=0)
        #         local_train_ldrs.append(local_train_ldr)
        local_train_ldrs = []
        for i in range(self.num_users):
            # 根據 iid 設定確定資料集和 num_workers
            if args.iid == 1:
                dataset = DatasetSplit(self.train_set, self.dict_users[i])
                num_workers = 2
            else:
                dataset = self.dict_users[i]
                num_workers = 0

            # 確定批次大小
            batch_size = self.batch_size
            if self.defense == 'instahide':
                batch_size = len(dataset)

            # 建立 DataLoader
            local_train_ldr = DataLoader(dataset, 
                                         batch_size=batch_size,
                                         shuffle=True, 
                                         num_workers=num_workers)
            local_train_ldrs.append(local_train_ldr)


        total_time=0
        file_name = "_".join(
                [ 'a',args.model_name, args.dataset,str(args.num_users),str(args.optim), str(args.lr_up), str(args.batch_size),  str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))])

        b=os.path.join(os.getcwd(), self.save_dir)
        if not os.path.exists(b):
            os.makedirs(b)
        fn=b+'/'+file_name+'.log'
        fn=file_name+'.log'
        fn=os.path.join(b,fn)
        print("training log saved in:",fn)

        lr_0=self.lr

        for epoch in range(self.epochs):

            global_state_dict=copy.deepcopy(self.model.state_dict())

            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)

            local_ws, local_losses,= [], []

            start = time.time()
            for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (self.epochs, self.lr)):

                self.model.load_state_dict(global_state_dict)

                # if need add add optimal beta here
                if "IB" in self.args.model_name:
                    self.model.beta = self.user_ib[idx]
                    # print(f"Now training client {idx} with beta {self.model.beta}")
                
                if self.args.defense == 'FedDPA':
                    # FedDPA's local_update returns model updates, not model weights
                    global_model_copy = copy.deepcopy(self.model)
                    local_w = self.defense_strategy.local_update(self.model, local_train_ldrs[idx], global_model_copy)
                    local_ws.append(local_w)
                    # FedDPA does not return loss, so we use a placeholder
                    local_losses.append(0.0)
                else:
                    local_w, local_loss = self.trainer._local_update_noback(local_train_ldrs[idx], self.local_ep, self.lr, self.optim, args.sampling_proportion)
                    
                    if self.args.defense == 'p2protect' and idx == 0 and (epoch+1) == self.epochs:
                        print(f"\nApplying P2Protect defense for client {idx}...")
                        x_client_list, y_client_list = [], []
                        for data, target in local_train_ldrs[idx]:
                            x_client_list.append(data)
                            y_client_list.append(target)
                        x_client = torch.cat(x_client_list, dim=0)
                        y_client = torch.cat(y_client_list, dim=0)

                        defended_model = defend_local_model(
                            original_model=self.model,
                            x_train=x_client,
                            y_train=y_client,
                            num_classes=self.num_classes,
                            eps_setting=self.args.p2protect_eps,
                            retrain_epochs=self.args.p2protect_retrain_epochs,
                            learning_rate=self.args.p2protect_lr,
                            batch_size=self.args.p2protect_batch_size,
                            device=self.device
                        )
                        local_w = copy.deepcopy(defended_model.state_dict())
                        print(f"P2Protect defense finished for client {idx}.")

                    elif self.args.defense in ['quant', 'sparse']:
                        model_grads = {}
                        for name, local_param in self.model.named_parameters():
                            if self.args.defense == 'quant':
                                model_grads[name]= local_w[name] - global_state_dict[name]
                                assert self.args.d_scale >= 1.0
                                model_grads[name]= quant(model_grads[name],int(self.args.d_scale))
                            elif self.args.defense == 'sparse':
                                model_grads[name]= local_w[name] - global_state_dict[name]
                                if model_grads[name].numel() > 1000:
                                    threshold = torch.topk( torch.abs(model_grads[name]).reshape(-1), int(model_grads[name].numel() * (1 - self.args.d_scale))).values[-1]
                                    model_grads[name]= torch.where(torch.abs(model_grads[name])<threshold, torch.zeros_like(model_grads[name]), model_grads[name])
                        
                        for key,value in model_grads.items():
                            if key in local_w:
                                local_w[key] = global_state_dict[key] + model_grads[key]
                    
                    local_ws.append(copy.deepcopy(local_w))
                    local_losses.append(local_loss)
                
                test_loss, test_kl_loss, test_ce_loss, test_acc=self.trainer.test(val_ldr)  

                if args.MIA_mode==1 and((epoch+1)%10==0 or epoch==0 or epoch in args.schedule_milestone or epoch-1 in args.schedule_milestone or epoch-2 in args.schedule_milestone)==1:
                    # Data that needs to be saved: the results of all clients for client0 and test set; client0 saves client0 data as train, and other clients as val
                    save_dict={}
                    save_dict['test_acc']=test_acc
                    save_dict['test_loss']=test_loss
                    crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
                    device = torch.device("cuda")

                    test_ldr_mia = DataLoader(self.test_set_mia, batch_size=self.batch_size , shuffle=False, num_workers=2)
                    test_res = get_all_losses(self.args, test_ldr_mia, self.model, crossentropy_noreduce, device)
                    save_dict['test_index']=self.testset_idx # 10000
                    save_dict['test_res']=test_res 

                    # target -> self.watch_train_client_id=0
                    train_res = get_all_losses_from_indexes(self.args, self.train_set_mia,self.train_idxs[self.watch_train_client_id], self.model)
                    save_dict['train_index']=self.train_idxs[self.watch_train_client_id]
                    save_dict['train_res']=train_res

                    # validation -> self.watch_val_client_id=1
                    val_res = get_all_losses_from_indexes(self.args, self.train_set_mia,self.train_idxs[self.watch_val_client_id], self.model)
                    save_dict['val_index']=self.train_idxs[self.watch_val_client_id]
                    save_dict['val_res']=val_res
                    
                    # mixed data 5000 test+ 500 * 9 --> 1000*1 test + 1000*9 other client 
                    mixed_indexs = []
                    needed_test_indexs = []
                    # data_num = int(len(self.train_idxs[1])/10)
                    if self.args.dataset == 'cifar100':
                        data_num = int(10000/self.num_users)
                        needed_test_indexs = random.sample(list(range(0,10000)), data_num)
                        # print('needed_test_indexs:',len(needed_test_indexs))
                        save_dict['needed_test_index']=needed_test_indexs
                    elif self.args.dataset == 'cifar10':
                        data_num = int(10000/self.num_users)
                        needed_test_indexs = random.sample(list(range(0,10000)), data_num)
                        save_dict['needed_test_index']=needed_test_indexs
                    elif self.args.dataset == 'dermnet':
                        data_num = 300
                        needed_test_indexs = None
                        # print('needed_test_indexs:',len(needed_test_indexs))
                    for c_id in range(1,self.num_users):
                        mixed_indexs.extend(random.sample(list(self.train_idxs[c_id]), data_num))
                    # print('len(mixed_indexs):', len(mixed_indexs))
                    # print('max:', max(mixed_indexs))
                    mix_res = get_all_losses_from_indexes(self.args, self.train_set_mia, mixed_indexs, self.model)
                    save_dict['mix_index']=mixed_indexs
                    save_dict['mix_res']=mix_res

                    if self.cosine_attack == True:# and idx == self.watch_train_client_id:

                        ## compute model grads
                        if self.args.defense == 'FedDPA':
                            # In FedDPA, local_w is the update (w_local - w_global), which is a list of tensors.
                            # The model gradient is approximated as w_global - w_local.
                            model_grads = torch.cat([-p.detach().cpu().flatten() for p in local_w])
                        else:
                            # In other cases, local_w is the full model state_dict.
                            model_grads= []
                            for name, local_param in self.model.named_parameters():
                                if local_param.requires_grad == True:
                                    # para_diff= local_w[name] - global_state_dict[name] # w2=w1-grad
                                    para_diff=  global_state_dict[name] - local_w[name] #0
                                    model_grads.append(para_diff.detach().cpu().flatten())
                            model_grads=torch.cat(model_grads,-1)

                        ## compute cosine score and grad diff score
                        if args.model_name == "ResNet18_IB_layer":
                            cos_model = models.__dict__[self.args.model_name](num_classes=self.num_classes, ib_layer_pos=self.args.ib_model_layer)
                        else:
                            cos_model = models.__dict__[self.args.model_name](num_classes=self.num_classes)
                        cos_model = cos_model.to(torch.device("cuda")) 
                        cos_model.load_state_dict(global_state_dict) # Load the basic global model
                        if "IB" in args.model_name:
                            train_cos,train_diffs, train_norm,val_cos, val_diffs,val_norm,test_cos, test_diffs,test_norm, mix_cos, mix_diffs,mix_norm=get_all_cos(self.args, cos_model, val_ldr,test_ldr_mia, self.test_set_mia, self.train_set_mia,
                                                                    self.train_idxs[self.watch_train_client_id],
                                                                    self.train_idxs[self.watch_val_client_id], 
                                                                    mixed_indexs,
                                                                    needed_test_indexs,
                                                                    model_grads, 
                                                                    self.lr, self.optim,
                                                                    self.model.beta)
                        else:
                            train_cos,train_diffs, train_norm,val_cos, val_diffs,val_norm,test_cos, test_diffs,test_norm, mix_cos, mix_diffs,mix_norm=get_all_cos(self.args, cos_model, val_ldr,test_ldr_mia, self.test_set_mia, self.train_set_mia,
                                                                    self.train_idxs[self.watch_train_client_id],
                                                                    self.train_idxs[self.watch_val_client_id], 
                                                                    mixed_indexs,
                                                                    needed_test_indexs,
                                                                    model_grads, 
                                                                    self.lr, self.optim)

                        save_dict['tarin_cos']=train_cos
                        save_dict['val_cos']=val_cos
                        save_dict['test_cos']=test_cos
                        save_dict['mix_cos']=mix_cos
                        save_dict['tarin_diffs']=train_diffs
                        save_dict['val_diffs']=val_diffs
                        save_dict['test_diffs']=test_diffs
                        save_dict['mix_diffs']=mix_diffs
                        save_dict['tarin_grad_norm']=train_norm
                        save_dict['val_grad_norm']=val_norm
                        save_dict['test_grad_norm']=test_norm
                        save_dict['mix_grad_norm']=mix_norm

                    # print out attack scores        
                    if not os.path.exists(os.path.join(os.getcwd(), self.save_dir)):
                        os.makedirs(os.path.join(os.getcwd(), self.save_dir))
                        print('MIA Score Saved in:', os.path.join(os.getcwd(), self.save_dir))
                    torch.save(save_dict, os.path.join(os.getcwd(), self.save_dir, f'client_{idx}_losses_epoch{epoch+1}.pkl'))
            if self.optim=="sgd":
                if self.args.lr_up=='common':
                    self.lr = self.lr * 0.99
                elif self.args.lr_up =='milestone':
                    if epoch in self.args.schedule_milestone:
                        self.lr *= 0.1
                else:
                    self.lr=lr_0 * (1 + math.cos(math.pi * epoch/ self.args.epochs)) / 2 
            else:
                pass

            client_weights = []
            for i in range(self.num_users):
                if args.iid == 1:
                    client_weight = len(DatasetSplit(self.train_set, self.dict_users[i])) / len(self.train_set)
                else:
                    client_weight = len(self.dict_users[i]) / len(self.train_set)
                client_weights.append(client_weight)
            
            self._fed_avg(local_ws, client_weights)
            self.model.load_state_dict(self.w_t)
            end = time.time()
            interval_time = end - start
            total_time+=interval_time

            # if (epoch + 1) % 10 == 0:
            #     f = os.path.join(os.getcwd(), self.save_dir, 'client_{}_losses_epoch{}.pkl')
            #     lira_score = lira_attack_ldh_cosine(f, (epoch+1), self.num_users, self.iid, attack_mode="cos")
            #     lira_loss_score = lira_attack_ldh_cosine(f, (epoch+1), self.num_users, self.iid, attack_mode="loss")
            #     for atk in ["cosine attack","grad diff","loss based","grad norm"]:
            #         cos_attack(f, self.num_users, (epoch+1), attack_mode=atk)

            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                loss_train_mean, loss_train_kl_mean, loss_train_ce_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, loss_val_kl_mean, loss_val_kl_mean, acc_val_mean = self.trainer.test(val_ldr)
                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean
                try:
                    loss_train_mean = loss_train_mean.item()
                    loss_val_mean = loss_val_mean.item()
                except:
                    loss_train_mean = loss_train_mean
                    loss_val_mean = loss_val_mean

                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))

                # use validation set as test set
                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                print('Epoch {}/{}  --time {:.1f}'.format(
                    epoch, self.epochs,
                    interval_time
                    )
                )

                print("Train Loss {:.4f} --- Train CE Loss {:.4f} --- Train KL Loss {:.4f} --- Val Loss {:.4f} --- "
                    .format(loss_train_mean, loss_train_ce_mean, loss_train_kl_mean, loss_val_mean))
                print("Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(acc_train_mean, acc_val_mean,
                                                                                                        self.logs[
                                                                                                            'best_test_acc']
                                                                                                        )
                    )
                s = 'epoch:{}, lr:{:.5f}, val_acc:{:.4f}, val_loss:{:.4f}, tarin_acc:{:.4f}, train_loss:{:.4f},time:{:.4f}, total_time:{:.4f}'.format(epoch,self.lr,acc_val_mean,loss_val_mean,acc_train_mean,loss_train_mean,interval_time,total_time)
                
                with open(fn,"a") as f:
                    json.dump({"epoch":epoch,"lr":round(self.lr,5),"train_acc":round(acc_train_mean,4  ),"train loss":round(loss_train_mean,4),"train ce loss":round(loss_train_ce_mean,4),"train kl loss":round(loss_train_kl_mean,4),
                               "test_acc":round(acc_val_mean,4),"test_loss":round(loss_val_mean,4),"time":round(total_time,2)},f)
                    f.write('\n')

        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f}  '.format(self.logs['best_test_loss'], 
                                                                                       self.logs['best_test_acc']
                                                                                       ))

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean

    def _fed_avg(self, local_updates, client_weights):
        if self.args.defense == 'FedDPA':
            # When using FedDPA, local_updates contains model updates (tensors)
            updated_global_model = self.defense_strategy.aggregate_updates(
                global_model=copy.deepcopy(self.model),
                client_updates=local_updates,
                client_weights=client_weights,
                noise_multiplier=self.args.noise_multiplier
            )
            self.w_t = updated_global_model.state_dict()
        else:
            # When not using FedDPA, local_updates contains full model weights (state_dicts)
            w_avg = copy.deepcopy(local_updates[0])
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] * client_weights[0]

                for i in range(1, len(local_updates)):
                    w_avg[k] += local_updates[i][k] * client_weights[i]

                self.w_t[k] = w_avg[k]


def get_loss_distributions(args, idx, MIA_trainset_dir,MIA_testloader, MIA_valset_dir, model):
        """ Obtain the member and nonmember loss distributions"""
        crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        device = torch.device("cuda")
        train_res = get_all_losses(args, MIA_trainset_dir[idx], model, crossentropy_noreduce, device)
        test_res = get_all_losses(args, MIA_testloader, model, crossentropy_noreduce, device)
        val_res = get_all_losses(args, MIA_valset_dir[idx], model, crossentropy_noreduce, device)
        return train_res,test_res,val_res

def get_all_losses(args, dataloader, model, criterion, device,req_logits=False):
    model.eval()
    losses = []
    logits = []
    labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            ### Forward
            if "IB" in args.model_name:
                outputs, kl_loss = model(inputs)
                ce_loss = criterion(outputs, targets)
                loss = kl_loss * model.beta + ce_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            ### Evaluate
            # loss = criterion(outputs, targets)
            losses.append(loss.cpu().numpy())
            logits.append(outputs.cpu())
            labels.append(targets.cpu())

    losses = np.concatenate(losses)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    return {"loss":losses,"logit":logits,"labels":labels}

def get_all_losses_from_indexes(args, dataset,indexes, model):
    criterion = nn.CrossEntropyLoss(reduction='none')
    device = torch.device("cuda")
    dataloader=DataLoader(DatasetSplit(dataset, indexes), batch_size = 200 ,shuffle=False, num_workers=0)
    model.eval()
    losses = []
    logits = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            ### Forward
            if "IB" in args.model_name:
                outputs, kl_loss = model(inputs)
                ce_loss = criterion(outputs, targets)
                loss = kl_loss * model.beta + ce_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            ### Evaluate
            # loss = criterion(outputs, targets)
            losses.append(loss.cpu().numpy())
            logits.append(outputs.cpu())
            labels.append(targets.cpu())

    losses = np.concatenate(losses)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    return {"loss":losses,"logit":logits,"labels":labels}

def get_all_cos(args, cos_model, initial_loader, test_dataloader, test_set, train_set, train_idxs, val_idxs, mix_idxs, needed_test_indexs, model_grads, lr, optim_choice, ib_beta=None): 
    device = torch.device("cuda")
    if optim_choice=="sgd":
        
        optimizer = optim.SGD(cos_model.parameters(),
                            lr,
                            momentum=0.9,
                            weight_decay=0.0005)
    else:
        optimizer = optim.AdamW(cos_model.parameters(),
                            lr,
                            weight_decay=0.0005)
    cos_models=[]
    if args.model_name not in ["ResNet18_IB_Block" ,"ResNet18_IB", "ResNet18_IB_layer"]:
        privacy_engine = PrivacyEngine()
        cos_model, optimizer, samples_loader = privacy_engine.make_private(
            module=cos_model,
            optimizer=optimizer,
            data_loader=initial_loader,
            noise_multiplier=0,
            max_grad_norm=1e10,
        )
 
    tarin_dataloader=DataLoader(DatasetSplit(train_set, train_idxs), batch_size = 10 ,shuffle=False, num_workers=4)
    # val_dataloader=DataLoader(DatasetSplit(train_set, val_idxs), batch_size = 10 ,shuffle=False, num_workers=4)
    test_dataloader=DataLoader(DatasetSplit(test_set, needed_test_indexs), batch_size=10 , shuffle=False, num_workers=4)
    mix_dataloader=DataLoader(DatasetSplit(train_set, mix_idxs), batch_size = 10 ,shuffle=False, num_workers=4)
    
    # start_train_time = time.time()
    # print(f"start train cos score")

    train_cos, train_diffs,train_norm=get_cos_score(args, tarin_dataloader,optimizer,cos_model,device,model_grads,ib_beta)

    # end_train_time = time.time()
    # print(f"End train cos score: {end_train_time-start_train_time}")

    test_cos, test_diffs,test_norm=get_cos_score(args, test_dataloader,optimizer,cos_model,device,model_grads,ib_beta)
    mix_cos, mix_diffs,mix_norm =get_cos_score(args, mix_dataloader,optimizer,cos_model,device,model_grads,ib_beta)

    # val_cos,val_diffs,val_norm=get_cos_score(val_dataloader,optimizer,cos_model,device,model_grads)
    val_cos,val_diffs,val_norm = None, None, None

    return train_cos, train_diffs, train_norm,val_cos,val_diffs,val_norm,test_cos, test_diffs,test_norm, mix_cos, mix_diffs,mix_norm

def get_cos_score(args,samples_ldr,optimizer,cos_model,device,model_grads,ib_beta=None):
     
    model_grads=model_grads.to(torch.device("cuda"))
    cos_model.train()  
    cos_scores=[] 
    grad_diffs=[]    
    sample_grads=[] 
    
    model_diff_norm=torch.norm(model_grads, p=2, dim=0)**2
    for batch_idx, (x, y) in enumerate(samples_ldr):
        sample_batch_grads=[]
        params = [p for p in cos_model.parameters() if p.requires_grad]
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        loss = torch.tensor(0.).to(device)

        if args.model_name in ["ResNet18_IB_Block", "ResNet18_IB", "ResNet18_IB_layer"]:
            pred, kl_loss = cos_model(x)
            ce_loss = F.cross_entropy(pred, y, reduction='none')
            ib_loss = kl_loss * ib_beta + ce_loss
            
            for i in range(x.size(0)):
                grads  = torch.autograd.grad(
                    outputs=ib_loss[i],
                    inputs=[p for p in cos_model.parameters() if p.requires_grad],
                    retain_graph=True,
                    create_graph=False,  # 記憶體較少
                    allow_unused=True
                )
                flat = torch.cat([
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(grads, params)
                ])

                sample_batch_grads.append(flat.unsqueeze(0)) 
            sample_batch_grads = torch.cat(sample_batch_grads, dim=0)
        
        else:
            if "IB" in args.model_name:
                pred, kl_loss = cos_model(x)
                ce_loss = F.cross_entropy(pred, y)
                ib_loss = kl_loss * ib_beta + ce_loss
                loss += ib_loss
            else:
                pred = cos_model(x)
                loss += F.cross_entropy(pred, y)
            
            loss.backward()

            sample_batch_grads=[]
            
            for name, param in cos_model.named_parameters(): #Save the grads of all parameters of the Model for the samples of the batch.
                if param.requires_grad==True:
                    #The i-th dimension is the grad of the parameter of the i-th sample
                    sample_batch_grads.append(param.grad_sample.flatten(start_dim=1))
                        

            sample_batch_grads=torch.cat(sample_batch_grads,1) # For each sample, concatenate its grads for all parameters into one line
        
        
        for sample_grad in sample_batch_grads:
            cos_score = F.cosine_similarity(sample_grad, model_grads, dim=0)
            cos_scores.append(cos_score)

            grad_diff=model_diff_norm - torch.norm(model_grads-sample_grad, p=2, dim=0)**2
            grad_diffs.append(grad_diff)

            sample_grads.append(torch.norm(sample_grad, p=2, dim=0)**2)

    return  torch.tensor(cos_scores).cpu(), torch.tensor(grad_diffs).cpu(), torch.tensor(sample_grads).cpu()

def main(args):
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,      
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,                
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'num_users': args.num_users
            }
            }
    save_dir = args.save_dir
    fl = FederatedLearning(args)

    logg, time, best_test_acc, test_acc = fl.train()                                         
                                             
    logs['net_info'] = logg 
    logs['test_acc'] = test_acc
    logs['bp_local'] = True if args.bp_interval == 0 else False

    if not os.path.exists(save_dir + args.model_name +'/' + args.dataset):
        os.makedirs(save_dir + args.model_name +'/' + args.dataset)
    torch.save(logs,
               save_dir + args.model_name +'/' + args.dataset + '/epoch_{}_E_{}_u_{}_{:.4f}_{:.4f}.pkl'.format(
                    args.epochs, args.local_ep, args.num_users, time, test_acc
               ))
    return

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

if __name__ == '__main__':
    args = parser_args()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    if "IB" in args.model_name:
        if args.model_name == "ResNet18_IB_layer" and args.iid != 2:
            args.save_dir=args.save_dir+'/'+f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_iblayer{args.ib_model_layer}_beta{args.ib_beta}_dynamic{args.dynamic_ib}_def{args.defense}_iid${args.iid}_${args.beta}_${args.optim}_local{args.local_ep}_s{args.seed}"
        elif args.iid == 2:
            args.save_dir=args.save_dir+'/'+f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_iblayer{args.ib_model_layer}_beta{args.ib_beta}_dynamic{args.dynamic_ib}_def{args.defense}_iid${args.iid}_nclass${args.n_classes}_${args.beta}_${args.optim}_local{args.local_ep}_s{args.seed}"
        else:
            args.save_dir=args.save_dir+'/'+f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_beta{args.ib_beta}_dynamic{args.dynamic_ib}_def{args.defense}_iid${args.iid}_${args.beta}_${args.optim}_local{args.local_ep}_s{args.seed}"
    else:
        if args.dp:
            args.save_dir=args.save_dir+'/'+f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_defDP_iid${args.iid}_${args.beta}_${args.optim}_local{args.local_ep}_s{args.seed}"
        else:
            args.save_dir=args.save_dir+'/'+f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_def{args.defense}_iid${args.iid}_${args.beta}_${args.optim}_local{args.local_ep}_s{args.seed}"
    print("scores saved in:",os.path.join(os.getcwd(), args.save_dir))
    args.log_folder_name=args.save_dir
    main(args)