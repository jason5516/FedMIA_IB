"""
簡單的 3 層 CNN 模型，用於 CIFAR10 分類
適用於互信息計算測試
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleCNN(nn.Module):
    """
    簡單的 3 層 CNN 模型
    
    架構：
    - Conv1: 3 -> 32 channels, 3x3 kernel, padding=1
    - Conv2: 32 -> 64 channels, 3x3 kernel, padding=1  
    - Conv3: 64 -> 128 channels, 3x3 kernel, padding=1
    - 每層後面跟 ReLU 和 MaxPool2d
    - 最後接全連接層進行分類
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        # 特徵提取層
        self.features = nn.Sequential(
            # 第一層卷積 (32x32x3 -> 32x32x32 -> 16x16x32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二層卷積 (16x16x32 -> 16x16x64 -> 8x8x64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層卷積 (8x8x64 -> 8x8x128 -> 4x4x128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分類器
        # 4x4x128 = 2048 個特徵
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # 特徵提取
        x = self.features(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分類
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def simple_cnn(**kwargs):
    """
    創建簡單的 3 層 CNN 模型
    
    Args:
        num_classes (int): 分類數量，默認為 10 (CIFAR10)
        dropout_rate (float): Dropout 比率，默認為 0.5
    
    Returns:
        SimpleCNN: 簡單 CNN 模型實例
    """
    return SimpleCNN(**kwargs)


def get_model_info(model):
    """
    獲取模型信息
    
    Args:
        model: PyTorch 模型
        
    Returns:
        dict: 包含模型參數數量等信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 假設 float32
    }


if __name__ == "__main__":
    # 測試模型
    print("=== 簡單 CNN 模型測試 ===")
    
    # 創建模型
    model = simple_cnn(num_classes=10)
    
    # 獲取模型信息
    info = get_model_info(model)
    print(f"總參數數量: {info['total_params']:,}")
    print(f"可訓練參數數量: {info['trainable_params']:,}")
    print(f"模型大小: {info['model_size_mb']:.2f} MB")
    
    # 測試前向傳播
    x = torch.randn(4, 3, 32, 32)  # 批次大小為 4 的 CIFAR10 輸入
    print(f"輸入形狀: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"輸出形狀: {output.shape}")
        print(f"輸出範例: {output[0]}")
    
    print("\n模型結構:")
    print(model)