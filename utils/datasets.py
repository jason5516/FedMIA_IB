import numpy as np
import json
import errno
import os
import sys
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_data(dataset, data_root, iid, num_users,data_aug, noniid_beta, save_path, n_class=10):
    ds = dataset 
    
    if ds == 'cifar10':
        if data_aug:
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])  
            transform_test = transforms.Compose([transforms.CenterCrop(32),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
            
            transform_train_mia = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            transform_test_mia = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        else:
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            transform_train_mia=transform_train
            transform_test_mia=transform_test

        train_set_mia = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train_mia
                                               )
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test_mia
                                                )

        
        
        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    
    if ds == 'cifar100':
        if data_aug :
            print("data_aug:",data_aug)
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),#
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(45),
                                                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])  
            transform_test = transforms.Compose([transforms.CenterCrop(32),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])
            
            transform_train_mia = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test_mia = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            
            transform_train_mia=transform_train
            transform_test_mia=transform_test

        train_set_mia = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train_mia
                                               )
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test_mia
                                                )

        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    if ds == 'dermnet':
        data=torch.load(data_root+"/dermnet_ts.pt")

        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        print(total_set[0].shape) # 19559, 3, 64, 64
        print(total_set[1].shape) # 19559
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:15000],total_set[1][0:15000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-4000:],total_set[1][-4000:] )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'oct':
        data=torch.load(data_root+"/oct_ts.pt")
        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:20000],total_set[1][0:20000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-2000:],total_set[1][-2000:] )

    if iid == 1:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA(train_set, num_users)

        return train_set, test_set, train_set_mia, test_set_mia, dict_users, train_idxs, val_idxs, None
    elif iid == 2 and ds == "cifar10":
        dict_users, train_idxs, val_idxs, client_size_map = cifar_all_class_num(train_set, n_class, num_users)
        client_label_distribution = {
            client_id: {int(class_id): int(count) for class_id, count in class_map.items()}
            for client_id, class_map in client_size_map.items()
        }
        
        with open(save_path + '/client_distribution.json', 'w') as f:
            json.dump(client_label_distribution, f, indent=4)
    elif iid == 3:
        dict_users, train_idxs, val_idxs, client_size_map = cifar_half_iid_half_noniid(train_set, num_users, noniid_beta)
        client_label_distribution = {
            client_id: {int(class_id): int(count) for class_id, count in class_map.items()}
            for client_id, class_map in client_size_map.items()
        }
        
        with open(save_path + '/client_distribution.json', 'w') as f:
            json.dump(client_label_distribution, f, indent=4)
    else:
        dict_users, train_idxs, val_idxs, client_size_map = cifar_beta(train_set, noniid_beta, num_users)

        client_label_distribution = {
            client_id: {int(class_id): int(count) for class_id, count in class_map.items()}
            for client_id, class_map in client_size_map.items()
        }
        
        with open(save_path + '/client_distribution.json', 'w') as f:
            json.dump(client_label_distribution, f, indent=4)

    return train_set, test_set, train_set_mia, test_set_mia, dict_users, train_idxs, val_idxs, client_label_distribution

def cifar_beta_correct(dataset, beta, n_clients):
    """
    Corrected version of cifar_beta that respects the DatasetSplit.
    """
    print("The dataset is splited with non-iid param ", beta)
    
    # Get labels for the indices in the current dataset split
    all_labels = np.array(dataset.dataset.targets)
    split_labels = all_labels[dataset.idxs]
    num_classes = len(dataset.dataset.classes)

    label_distributions = []
    for y in range(num_classes):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    client_idx_map = {i: {} for i in range(n_clients)}
    client_size_map = {i: {} for i in range(n_clients)}

    # Group indices from the split by class
    class_indices_in_split = {c: [] for c in range(num_classes)}
    for i, idx in enumerate(dataset.idxs):
        label = split_labels[i]
        class_indices_in_split[label].append(idx)

    for y in range(num_classes):
        label_y_idx = np.array(class_indices_in_split[y])
        label_y_size = len(label_y_idx)

        if label_y_size == 0:
            for i in range(n_clients):
                client_size_map[i][y] = 0
                client_idx_map[i][y] = []
            continue

        sample_size = (label_distributions[y] * label_y_size).astype(np.int32)
        sample_size[n_clients - 1] += label_y_size - np.sum(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            start = sample_interval[i - 1] if i > 0 else 0
            client_idx_map[i][y] = label_y_idx[start:sample_interval[i]]

    train_idxs = []
    val_idxs = []
    client_datasets = []
    all_idxs = dataset.idxs # Use indices from the split for validation set calculation
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values())).astype(int)
        np.random.shuffle(client_i_idx)
        
        # The subset should be from the original dataset, using the correct indices
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)
        
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs) - set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs, client_size_map

def cifar_half_iid_half_noniid(dataset, num_users, noniid_beta):
    
    num_iid_users = num_users // 2
    num_noniid_users = num_users - num_iid_users

    # Split dataset into two halves
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    iid_indices = indices[:total_size // 2]
    noniid_indices = indices[total_size // 2:]

    iid_dataset = DatasetSplit(dataset.dataset, iid_indices)
    noniid_dataset = DatasetSplit(dataset.dataset, noniid_indices)

    # IID part
    dict_users_iid, train_idxs_iid, val_idxs_iid = cifar_iid_MIA(iid_dataset, num_iid_users)

    # Non-IID part
    _, train_idxs_noniid, val_idxs_noniid, client_size_map_noniid = cifar_beta_correct(noniid_dataset, noniid_beta, num_noniid_users)

    dict_users = {}
    train_idxs = []
    val_idxs = []
    client_size_map = {}

    # Combine IID results
    for i in range(num_iid_users):
        # Remap iid indices back to original dataset indices
        original_train_indices = {iid_indices[j] for j in dict_users_iid[i]}
        dict_users[i] = list(original_train_indices)
        train_idxs.append(list(original_train_indices))
        # For val_idxs in iid, we can take all other indices from the original dataset
        val_idxs.append(list(set(range(total_size)) - original_train_indices))


    # Combine Non-IID results
    dict_users_noniid = {}
    for i in range(num_noniid_users):
        user_id = i + num_iid_users
        # No remapping needed as cifar_beta_correct returns original indices
        original_indices = train_idxs_noniid[i]
        dict_users_noniid[user_id] = list(original_indices)
        train_idxs.append(list(original_indices))
        # For val_idxs in non-iid, we can take all other indices
        val_idxs.append(list(set(range(total_size)) - set(original_indices)))

    dict_users.update(dict_users_noniid)

    # Calculate client_size_map for all clients
    labels = np.array(dataset.dataset.targets)
    for i in range(num_users):
        client_indices = np.array(list(dict_users[i]))
        if len(client_indices) == 0:
            class_counts = {c: 0 for c in range(len(dataset.dataset.classes))}
        else:
            client_labels = labels[client_indices]
            class_counts = {c: 0 for c in range(len(dataset.dataset.classes))}
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                class_counts[label] = count
        client_size_map[i] = class_counts

    return dict_users, train_idxs, val_idxs, client_size_map


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        image, label = self.dataset[self.idxs[item]]
        return image, label

class WMDataset(Dataset):
    def __init__(self, root, labelpath, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.labelpath = labelpath
        self.labels = np.loadtxt(self.labelpath)
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)

def prepare_wm(datapath='/trigger/pics/', num_back=1, shuffle=True):
    
    triggerroot = datapath
    labelpath = '/home/lbw/Data/trigger/labels-cifar.txt'

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ]

    transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset(triggerroot, labelpath, wm_transform)
    
    dict_users_back = wm_iid(dataset, num_back, 100)

    return dataset, dict_users_back

def prepare_wm_indistribution(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    triggerroot = datapath
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.ToTensor()
    ]

    #transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset_indistribution(triggerroot, wm_transform)
    
    num_all = num_trigger * num_back 

    dataset = DatasetSplit(dataset, np.arange(0, num_all))
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back

def prepare_wm_new(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    wm_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(datapath, wm_transform)
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back



class WMDataset_indistribution(Dataset):
    def __init__(self, root, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = 5
        if index in self.cache:
            img = self.cache[index]
        else:
        
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)
