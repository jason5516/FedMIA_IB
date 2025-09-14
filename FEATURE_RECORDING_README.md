# 特徵值記錄功能說明

## 功能概述

本功能在原有的聯邦學習程式碼基礎上，添加了每個client模型每層layer輸出特徵值以及輸入資料X與label Y的記錄功能，每5個global round記錄一次，每個client限制記錄200筆資料。

## 主要特性

- **自動記錄**: 每5個global round自動觸發特徵值記錄
- **限制數量**: 每個client限制記錄200筆資料的特徵值
- **完整記錄**: 記錄每層的輸出特徵值、輸入資料X和標籤Y
- **記憶體優化**: 使用forward hooks機制，避免修改模型結構
- **無侵入性**: 不影響原有的訓練邏輯和其他功能

## 新增的程式碼組件

### 1. 特徵值記錄相關屬性
在 `FederatedLearning` 類的 `__init__` 方法中添加：
```python
# 添加特徵值記錄相關屬性
self.feature_hooks = {}
self.layer_features = {}
self.record_features = False
self.current_client_id = None
self.current_input_data = None
self.current_labels = None
```

### 2. Hook機制方法
- `register_feature_hooks(model)`: 註冊forward hooks來捕獲每層的輸出特徵值
- `remove_feature_hooks()`: 移除所有的forward hooks
- `save_layer_features_and_data()`: 保存層特徵值、輸入資料和標籤

### 3. 特殊訓練函數
- `_local_update_with_feature_recording()`: 帶有特徵值記錄的本地訓練函數

## 記錄的資料格式

每個記錄文件包含以下資料：
```python
{
    'epoch': int,                    # 當前epoch
    'client_id': int,               # 客戶端ID
    'input_data': numpy.ndarray,    # 輸入資料X
    'labels': numpy.ndarray,        # 標籤Y
    'layer_features': {             # 每層的特徵值
        'layer_name': [             # 層名稱
            numpy.ndarray,          # 該層在不同batch的輸出特徵值
            ...
        ],
        ...
    }
}
```

## 文件保存位置

特徵值記錄文件保存在：
```
{save_dir}/layer_features/client_{client_id}_features_epoch_{epoch}.pkl
```

例如：
- `./results/layer_features/client_0_features_epoch_5.pkl`
- `./results/layer_features/client_1_features_epoch_5.pkl`
- `./results/layer_features/client_0_features_epoch_10.pkl`

## 使用方法

### 1. 正常運行訓練
```bash
python main.py --model_name ResNet18 --dataset cifar10 --epochs 20 --num_users 10
```

### 2. 檢查記錄文件
訓練完成後，檢查 `{save_dir}/layer_features/` 目錄下的 `.pkl` 文件。

### 3. 載入和分析記錄的資料
```python
import torch
import numpy as np

# 載入記錄文件
data = torch.load('path/to/client_0_features_epoch_5.pkl', map_location='cpu')

# 查看基本資訊
print(f"Epoch: {data['epoch']}")
print(f"Client ID: {data['client_id']}")
print(f"Input data shape: {data['input_data'].shape}")
print(f"Labels shape: {data['labels'].shape}")

# 查看層特徵值
layer_features = data['layer_features']
for layer_name, features_list in layer_features.items():
    print(f"Layer {layer_name}: {len(features_list)} batches")
    if features_list:
        print(f"  Feature shape: {features_list[0].shape}")
```

## 測試功能

運行測試腳本來驗證功能：
```bash
python test_feature_recording.py
```

測試腳本會：
1. 運行一個簡化的聯邦學習訓練
2. 檢查是否正確生成了特徵值記錄文件
3. 驗證文件內容的完整性

## 記錄頻率和數量限制

- **記錄頻率**: 每5個global round記錄一次
- **數量限制**: 每個client限制記錄200筆資料
- **可修改頻率**: 在 `train()` 方法中修改條件 `(epoch + 1) % 5 == 0`
- **可修改數量**: 在 `_local_update_with_feature_recording()` 和 `register_feature_hooks()` 中修改 `max_samples_to_record` 和 `max_feature_samples`

例如，改為每3個round記錄一次：
```python
should_record_features = (epoch + 1) % 3 == 0
```

例如，改為每個client記錄500筆資料：
```python
max_samples_to_record = 500  # 在 _local_update_with_feature_recording 中
self.max_feature_samples = 500  # 在 register_feature_hooks 中
```

## 注意事項

1. **記憶體使用**: 特徵值記錄會增加記憶體使用，但已限制為200筆資料以控制記憶體消耗
2. **存儲空間**: 每個記錄文件包含200筆資料的特徵值，文件大小相對可控
3. **訓練時間**: 記錄過程會略微增加訓練時間，但影響有限
4. **相容性**: 與現有的防禦機制（如DP、FedDPA等）相容
5. **數據一致性**: 輸入資料X、標籤Y和各層特徵值的樣本數量保持一致（都是200筆）

## 故障排除

### 1. 記憶體不足
- 減少batch size
- 減少記錄頻率
- 只記錄特定層的特徵值

### 2. 沒有生成記錄文件
- 檢查 `save_dir` 權限
- 確認訓練運行了足夠的epochs
- 檢查控制台輸出中的錯誤訊息

### 3. 文件損壞
- 檢查磁碟空間是否充足
- 確認訓練過程沒有被中斷

## 擴展功能

可以根據需要進一步擴展：
1. 只記錄特定層的特徵值
2. 添加特徵值統計資訊（均值、方差等）
3. 支援不同的記錄格式（HDF5、NPZ等）
4. 添加資料壓縮功能