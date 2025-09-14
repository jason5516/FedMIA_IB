#!/usr/bin/env python3
"""
測試MI計算功能的簡單腳本
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 添加路徑以便導入模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from information.mutual_information_calculator import MutualInformationCalculator

def create_simple_model():
    """創建一個簡單的測試模型"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.layer1 = nn.Linear(784, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 64)
            self.layer4 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x1 = self.relu(self.layer1(x))
            x2 = self.relu(self.layer2(x1))
            x3 = self.relu(self.layer3(x2))
            x4 = self.layer4(x3)
            return x4
    
    return SimpleModel()

def test_mi_calculator():
    """測試MI計算器"""
    print("開始測試MI計算器...")
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 創建MI計算器
    mi_calculator = MutualInformationCalculator(device=device, num_layers=4)
    
    # 創建測試資料
    batch_size = 32
    input_dim = 784  # 28x28 for MNIST-like data
    num_classes = 10
    
    # 模擬輸入資料和標籤
    x = torch.randn(batch_size, input_dim).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 模擬層輸出
    layer1_output = torch.randn(batch_size, 256).to(device)
    layer2_output = torch.randn(batch_size, 128).to(device)
    layer3_output = torch.randn(batch_size, 64).to(device)
    layer4_output = torch.randn(batch_size, num_classes).to(device)
    
    layer_outputs = [layer1_output, layer2_output, layer3_output, layer4_output]
    
    try:
        # 計算MI
        print("計算互信息...")
        I_XT, I_TY = mi_calculator.compute_mutual_information(x, y, layer_outputs, epoch=0)
        
        print("MI計算成功！")
        print(f"I(X;T) 形狀: {I_XT.shape}")
        print(f"I(T;Y) 形狀: {I_TY.shape}")
        print(f"I(X;T) 值: {I_XT.detach().cpu().numpy()}")
        print(f"I(T;Y) 值: {I_TY.detach().cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"MI計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hook_mechanism():
    """測試hook機制"""
    print("\n開始測試hook機制...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_simple_model().to(device)
    
    # 用於存儲層輸出的列表
    layer_features = []
    
    def create_hook(layer_name):
        def hook_fn(module, input, output):
            print(f"Hook觸發: {layer_name}, 輸出形狀: {output.shape}")
            layer_features.append(output.detach())
        return hook_fn
    
    # 註冊hooks
    hooks = []
    for name, module in model.named_modules():
        if name in ['layer1', 'layer2', 'layer3', 'layer4']:
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
    
    # 創建測試資料
    batch_size = 16
    x = torch.randn(batch_size, 1, 28, 28).to(device)  # MNIST-like input
    
    try:
        # 前向傳播
        print("執行前向傳播...")
        with torch.no_grad():
            output = model(x)
        
        print(f"模型輸出形狀: {output.shape}")
        print(f"捕獲的層特徵數量: {len(layer_features)}")
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return True
        
    except Exception as e:
        print(f"Hook測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("測試MI計算功能")
    print("=" * 60)
    
    # 測試MI計算器
    mi_test_result = test_mi_calculator()
    
    # 測試hook機制
    hook_test_result = test_hook_mechanism()
    
    print("\n" + "=" * 60)
    print("測試結果總結:")
    print(f"MI計算器測試: {'通過' if mi_test_result else '失敗'}")
    print(f"Hook機制測試: {'通過' if hook_test_result else '失敗'}")
    
    if mi_test_result and hook_test_result:
        print("所有測試通過！修改後的代碼應該可以正常工作。")
    else:
        print("部分測試失敗，需要進一步調試。")
    print("=" * 60)