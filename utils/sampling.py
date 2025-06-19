import numpy as np
from numpy import random
import numpy as np
import torch
from torch.utils.data import Subset


np.random.seed(1)

def wm_iid(dataset, num_users, num_back):
    """
    Sample I.I.D. client data from watermark dataset
    """
    num_items = min(num_back, int(len(dataset)/num_users))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid_MIA(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idx0=all_idxs
    train_idxs=[]
    val_idxs=[]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        train_idxs.append(list(dict_users[i] ))
        all_idxs = list(set(all_idxs) - dict_users[i])
        val_idxs.append(list(set(all_idx0)-dict_users[i]))
    return dict_users, train_idxs, val_idxs

def cifar_all_class_num(
    dataset,
    classes_per_client,
    num_users,
    val_size=5000,
    random_seed=42
):
    np.random.seed(random_seed)
    total_samples = len(dataset.dataset)
    # 每个 client 的目标样本总数
    samples_per_client = total_samples // num_users // 2

    # 所有标签 & 类别数
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    num_classes = len(np.unique(labels))

    # 打乱类别顺序，round-robin 分配 classes_per_client 类给每个 client
    class_order = np.arange(num_classes)
    np.random.shuffle(class_order)
    client_classes = {}
    for i in range(num_users):
        start = (i * classes_per_client) % num_classes
        cls = class_order[start:start + classes_per_client].tolist()
        if len(cls) < classes_per_client:
            cls += class_order[:(classes_per_client - len(cls))].tolist()
        client_classes[i] = cls

    # 每个类的所有索引
    class_indices = {c: np.where(labels == c)[0].tolist()
                     for c in range(num_classes)}

    client_datasets = []
    train_idxs = []
    val_idxs = []
    client_size_map = {}

    for client_id, classes in client_classes.items():
        base = samples_per_client // classes_per_client
        rem  = samples_per_client % classes_per_client

        idxs = []
        counts = {}
        for j, c in enumerate(classes):
            need = base + (1 if j < rem else 0)
            pool = class_indices[c]
            replace = len(pool) < need
            chosen = np.random.choice(pool, size=need, replace=replace).tolist()
            idxs.extend(chosen)
            counts[c] = need

        np.random.shuffle(idxs)
        train_idxs.append(idxs)
        client_size_map[client_id] = counts
        client_datasets.append(Subset(dataset.dataset, idxs))
    
    all_train = set(sum(train_idxs, []))
    remaining = list(set(range(total_samples)) - all_train)
    np.random.shuffle(remaining)

    # 3) 从剩余中分层抽样做共享验证集
    base_val = val_size // num_classes
    rem_val  = val_size % num_classes

    # 剩余类别索引
    rem_by_class = {c: [] for c in range(num_classes)}
    for idx in remaining:
        rem_by_class[labels[idx]].append(idx)

    val_idxs = []
    val_size_map = {}
    for c in range(num_classes):
        need = base_val + (1 if c < rem_val else 0)
        pool = rem_by_class[c]
        replace = len(pool) < need
        chosen = np.random.choice(pool, size=need, replace=replace).tolist()
        val_idxs.extend(chosen)
        val_size_map[c] = need

    np.random.shuffle(val_idxs)

    client_val_idxs = []
    for c in range(num_users):
        client_val_idxs.append(val_idxs)


    return client_datasets, train_idxs, client_val_idxs, client_size_map

def cifar_class_num(dataset, n_class, num_users, num_classes=10):
    # reproducibility
        # 全量索引
    all_idx0 = np.arange(len(dataset))
    all_idxs = all_idx0.copy().tolist()
    dict_users = {}
    train_idxs = []
    val_idxs = []
    
    # I.I.D. 分配
    num_items = len(dataset) // num_users
    for i in range(num_users):
        chosen = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = set(chosen)
        train_idxs.append(list(chosen))
        all_idxs = list(set(all_idxs) - dict_users[i])
        val_idxs.append(list(set(all_idx0) - dict_users[i]))
    
    # 只保留 user 0 的 n_class 類別
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    selected_classes = np.random.choice(np.arange(num_classes), size=n_class, replace=False)
    user0_idx = np.array(list(dict_users[0]))
    mask = np.isin(labels[user0_idx], selected_classes)
    filtered0 = user0_idx[mask]
    dict_users[0] = set(filtered0.tolist())
    train_idxs[0] = filtered0.tolist()
    val_idxs[0] = list(set(all_idx0) - dict_users[0])
    
    # 計算每個 client 在各 class 的大小
    client_size_map = {i: {} for i in range(num_users)}
    for i in range(num_users):
        idxs = np.array(list(dict_users[i]))
        if idxs.size == 0:
            # 沒有資料就全部歸零
            for c in range(num_classes):
                client_size_map[i][c] = 0
        else:
            # 計算各類別出現次數
            classes, counts = np.unique(labels[idxs], return_counts=True)
            # 初始化為 0，再填入非零類別
            for c in range(num_classes):
                client_size_map[i][c] = int(counts[classes.tolist().index(c)]) if c in classes else 0

    return dict_users, train_idxs, val_idxs, client_size_map


def cifar_beta(dataset, beta, n_clients):  
     #beta = 0.1, n_clients = 10
    print("The dataset is splited with non-iid param ", beta)
    label_distributions = []
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    #print("labels:",labels)
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}
    #print("classes:",dataset.dataset.classes)
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_y_idx = np.where(labels == y)[0] # [93   107   199   554   633   639 ... 54222]
        label_y_size = len(label_y_idx)
        #print(label_y_idx[0:100])
        
        sample_size = (label_distributions[y]*label_y_size).astype(np.int32)
        #print(sample_size)
        sample_size[n_clients-1] += label_y_size - np.sum(sample_size)
        #print(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i>0 else 0):sample_interval[i]]
    
    train_idxs=[]
    val_idxs=[]    
    client_datasets = []
    all_idxs=[i for i in range(len(dataset))]
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)
        # save the idxs for attack
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs)-set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs, client_size_map

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

