#!/usr/bin/env python3
"""
簡單 3 層 CNN CIFAR10 訓練與互信息計算測試腳本

此腳本使用簡單的 3 層 CNN 模型在 CIFAR10 資料集上進行訓練，
並在每個 iteration 中使用 hook 提取最後一層特徵，
計算互信息(MI)並記錄 IXZ 和 IZY 值。
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.simple_cnn import simple_cnn
from models.resnet_cifar import ResNet18
from information.mutual_information_calculator import MutualInformationCalculator


class MITrainer:
    """簡單 CNN 模型訓練器，包含互信息計算功能"""
    
    def __init__(self, device='cuda', save_dir='./results'):
        self.device = device
        self.save_dir = save_dir
        self.setup_directories()
        
        # 初始化互信息計算器
        self.mi_calculator = MutualInformationCalculator(device=device)
        
        # 記錄相關變數
        self.layer_features = []
        self.mi_records = {'IXZ': [], 'IZY': [], 'iteration': []}
        self.training_records = {'loss': [], 'accuracy': [], 'iteration': []}
        
        # Hook相關變數
        self.hook_handle = None
        self.current_features = None
        
        print(f"初始化簡單CNN MI訓練器，使用設備: {device}")
        print(f"結果將保存到: {save_dir}")
    
    def setup_directories(self):
        """創建必要的目錄"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)
        
    def load_cifar10_data(self, batch_size=128, data_augmentation=True):
        """載入CIFAR10資料集"""
        print("載入CIFAR10資料集...")
        
        if data_augmentation:
            # 訓練時使用資料增強
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        # 測試時不使用資料增強
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 載入資料集
        trainset = torchvision.datasets.CIFAR10(
            root='../Data', train=True, download=True, transform=transform_train
        )
        self.train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        
        testset = torchvision.datasets.CIFAR10(
            root='../Data', train=False, download=True, transform=transform_test
        )
        self.test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        print(f"訓練集大小: {len(trainset)}, 測試集大小: {len(testset)}")
        print(f"批次大小: {batch_size}")
        
    def create_model(self):
        """創建簡單 CNN 模型"""
        print("創建簡單 3 層 CNN 模型...")
        self.model = ResNet18(num_classes=10)
        self.model = self.model.to(self.device)
        
        # 計算模型參數數量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型總參數數量: {total_params:,}")
        print(f"可訓練參數數量: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / (1024 * 1024):.2f} MB")
        
        return self.model
    
    def register_hook(self):
        """註冊hook來捕獲最後一層的特徵"""
        def hook_fn(module, input, output):
            """Hook函數，捕獲層的輸出特徵"""
            # 將特徵轉換為CPU並detach，避免記憶體問題
            if isinstance(output, tuple):
                features = output[0].detach().cpu()
            else:
                features = output.detach().cpu()
            
            # 如果是4D張量(卷積層輸出)，需要flatten
            if len(features.shape) == 4:
                features = features.view(features.size(0), -1)
            
            self.current_features = features
        
        # 對於簡單CNN，我們選擇features層的輸出（卷積特徵）
        self.hook_handle = self.model.layer1.register_forward_hook(hook_fn)
        
        print("已註冊hook到features層")
    
    def remove_hook(self):
        """移除hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("已移除hook")
    
    def compute_mi_for_batch(self, x, y, iteration):
        """為當前批次計算互信息"""
        try:
            # 確保模型在評估模式下進行MI計算
            self.model.eval()
            
            with torch.no_grad():
                # 前向傳播以觸發hook
                _ = self.model(x)
                
                # 獲取捕獲的特徵
                if self.current_features is not None:
                    # 準備資料
                    x_cpu = x.detach().cpu()
                    y_cpu = y.detach().cpu()
                    features_cpu = self.current_features
                    
                    # 將輸入資料flatten
                    x_flat = x_cpu.view(x_cpu.size(0), -1)
                    
                    # 計算互信息
                    layer_outputs = [features_cpu]  # 只有一層特徵
                    I_XT, I_TY = self.mi_calculator.compute_mutual_information(
                        x_flat, y_cpu, layer_outputs, epoch=iteration
                    )
                    
                    # 記錄結果
                    ixz_value = I_XT[0].item() if len(I_XT) > 0 else 0.0
                    izy_value = I_TY[0].item() if len(I_TY) > 0 else 0.0
                    
                    self.mi_records['IXZ'].append(ixz_value)
                    self.mi_records['IZY'].append(izy_value)
                    self.mi_records['iteration'].append(iteration)
                    
                    return ixz_value, izy_value
                else:
                    print("警告: 未捕獲到特徵")
                    return 0.0, 0.0
                    
        except Exception as e:
            print(f"計算MI時發生錯誤: {e}")
            return 0.0, 0.0
        finally:
            # 恢復訓練模式
            self.model.train()
    
    def train_epoch(self, optimizer, criterion, epoch, total_iterations):
        """訓練一個epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向傳播
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 統計
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 記錄訓練指標
            current_iteration = total_iterations + batch_idx
            self.training_records['loss'].append(loss.item())
            self.training_records['accuracy'].append(100. * correct / total)
            self.training_records['iteration'].append(current_iteration)
            
            # 計算互信息（每個batch）
            ixz, izy = self.compute_mi_for_batch(data, target, current_iteration)
            
            # 更新進度條
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'IXZ': f'{ixz:.4f}',
                'IZY': f'{izy:.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, total_iterations + len(self.train_loader)
    
    def test_model(self):
        """測試模型性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def save_results(self):
        """保存訓練結果和MI記錄"""
        print("保存結果...")
        
        # 保存MI記錄
        mi_data = {
            'IXZ': self.mi_records['IXZ'],
            'IZY': self.mi_records['IZY'],
            'iteration': self.mi_records['iteration']
        }
        
        with open(os.path.join(self.save_dir, 'mi_records.json'), 'w') as f:
            json.dump(mi_data, f, indent=2)
        
        # 保存訓練記錄
        training_data = {
            'loss': self.training_records['loss'],
            'accuracy': self.training_records['accuracy'],
            'iteration': self.training_records['iteration']
        }
        
        with open(os.path.join(self.save_dir, 'training_records.json'), 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # 保存模型
        torch.save(self.model.state_dict(), 
                  os.path.join(self.save_dir, 'models', 'simple_cnn_final.pth'))
        
        print(f"結果已保存到 {self.save_dir}")
    
    def plot_information_plane(self):
        """繪製 Information Plane (IP) - 用顏色深淺表示 iteration"""
        print("繪製 Information Plane...")
        
        if len(self.mi_records['IXZ']) == 0 or len(self.mi_records['IZY']) == 0:
            print("警告: 沒有足夠的MI資料來繪製Information Plane")
            return
        
        # 準備資料
        ixz_values = np.array(self.mi_records['IXZ'])
        izy_values = np.array(self.mi_records['IZY'])
        iterations = np.array(self.mi_records['iteration'])
        
        # 正規化 iteration 值用於顏色映射 (0-1)
        if len(iterations) > 1:
            norm_iterations = (iterations - iterations.min()) / (iterations.max() - iterations.min())
        else:
            norm_iterations = np.array([0.5])
        
        # 創建 Information Plane 圖
        plt.figure(figsize=(12, 10))
        
        # 主要的 Information Plane 散點圖
        plt.subplot(2, 2, (1, 2))  # 佔據上方兩個位置
        scatter = plt.scatter(ixz_values, izy_values, c=norm_iterations,
                            cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 添加軌跡線
        plt.plot(ixz_values, izy_values, 'k-', alpha=0.3, linewidth=1, zorder=1)
        
        # 標記起始點和結束點
        if len(ixz_values) > 0:
            plt.scatter(ixz_values[0], izy_values[0], c='red', s=100, marker='o',
                       label='Start', edgecolors='black', linewidth=2, zorder=5)
            plt.scatter(ixz_values[-1], izy_values[-1], c='blue', s=100, marker='s',
                       label='End', edgecolors='black', linewidth=2, zorder=5)
        
        plt.xlabel('I(X;Z) - Input-Feature Mutual Information', fontsize=14)
        plt.ylabel('I(Z;Y) - Feature-Output Mutual Information', fontsize=14)
        plt.title('Information Plane - Training Trajectory\n(顏色深淺表示訓練進度)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加顏色條
        cbar = plt.colorbar(scatter)
        cbar.set_label('Training Progress (Normalized Iteration)', fontsize=12)
        
        # 左下角: I(X;Z) 隨時間變化
        plt.subplot(2, 2, 3)
        plt.plot(iterations, ixz_values, 'r-', linewidth=2, alpha=0.8)
        plt.scatter(iterations, ixz_values, c=norm_iterations, cmap='viridis', s=20, alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('I(X;Z)')
        plt.title('I(X;Z) vs Iteration')
        plt.grid(True, alpha=0.3)
        
        # 右下角: I(Z;Y) 隨時間變化
        plt.subplot(2, 2, 4)
        plt.plot(iterations, izy_values, 'b-', linewidth=2, alpha=0.8)
        plt.scatter(iterations, izy_values, c=norm_iterations, cmap='viridis', s=20, alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('I(Z;Y)')
        plt.title('I(Z;Y) vs Iteration')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'information_plane.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 創建單獨的大型 Information Plane 圖
        plt.figure(figsize=(10, 8))
        
        # 使用更精細的顏色映射
        scatter = plt.scatter(ixz_values, izy_values, c=norm_iterations,
                            cmap='plasma', s=80, alpha=0.8, edgecolors='white', linewidth=1)
        
        # 添加軌跡線，使用漸變效果
        for i in range(len(ixz_values)-1):
            plt.plot([ixz_values[i], ixz_values[i+1]], [izy_values[i], izy_values[i+1]],
                    'k-', alpha=0.2 + 0.6*norm_iterations[i], linewidth=2)
        
        # 標記重要點
        if len(ixz_values) > 0:
            plt.scatter(ixz_values[0], izy_values[0], c='lime', s=150, marker='o',
                       label='Training Start', edgecolors='black', linewidth=2, zorder=10)
            plt.scatter(ixz_values[-1], izy_values[-1], c='red', s=150, marker='s',
                       label='Training End', edgecolors='black', linewidth=2, zorder=10)
        
        plt.xlabel('I(X;Z) - Input-Feature Mutual Information', fontsize=16)
        plt.ylabel('I(Z;Y) - Feature-Output Mutual Information', fontsize=16)
        plt.title('Information Plane - Neural Network Training Dynamics\n深度學習資訊理論視角', fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # 添加顏色條
        cbar = plt.colorbar(scatter)
        cbar.set_label('Training Progress (淺→深 = 早期→後期)', fontsize=14)
        
        # 添加統計資訊
        if len(ixz_values) > 0:
            plt.text(0.02, 0.98, f'Total Iterations: {len(ixz_values)}\n'
                               f'I(X;Z) Range: [{ixz_values.min():.3f}, {ixz_values.max():.3f}]\n'
                               f'I(Z;Y) Range: [{izy_values.min():.3f}, {izy_values.max():.3f}]',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'information_plane_detailed.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Information Plane 圖表已保存")

    def plot_results(self):
        """繪製訓練結果和MI曲線"""
        print("繪製結果圖表...")
        
        # 創建子圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Simple CNN CIFAR10 Training Results with Mutual Information', fontsize=16)
        
        # 訓練損失
        axes[0, 0].plot(self.training_records['iteration'], self.training_records['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 訓練準確率
        axes[0, 1].plot(self.training_records['iteration'], self.training_records['accuracy'])
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True)
        
        # IXZ (I(X;Z))
        axes[1, 0].plot(self.mi_records['iteration'], self.mi_records['IXZ'], 'r-', label='I(X;Z)')
        axes[1, 0].set_title('Mutual Information I(X;Z)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('I(X;Z)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # IZY (I(Z;Y))
        axes[1, 1].plot(self.mi_records['iteration'], self.mi_records['IZY'], 'b-', label='I(Z;Y)')
        axes[1, 1].set_title('Mutual Information I(Z;Y)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('I(Z;Y)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'training_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 單獨繪製MI曲線
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.mi_records['iteration'], self.mi_records['IXZ'], 'r-', linewidth=2)
        plt.title('I(X;Z) - Input-Feature Mutual Information', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('I(X;Z)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.mi_records['iteration'], self.mi_records['IZY'], 'b-', linewidth=2)
        plt.title('I(Z;Y) - Feature-Output Mutual Information', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('I(Z;Y)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'mutual_information.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 繪製 Information Plane
        self.plot_information_plane()
        
        print("圖表已保存")
    
    def train(self, epochs=10, lr=0.001, weight_decay=5e-4):
        """主要訓練函數"""
        print(f"開始訓練 {epochs} 個epochs...")
        
        # 設置優化器和損失函數
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # 註冊hook
        self.register_hook()
        
        total_iterations = 0
        
        try:
            for epoch in range(epochs):
                print(f"\n=== Epoch {epoch+1}/{epochs} ===")
                
                # 訓練一個epoch
                epoch_loss, epoch_acc, total_iterations = self.train_epoch(
                    optimizer, criterion, epoch, total_iterations
                )
                
                # 測試模型
                test_loss, test_acc = self.test_model()
                
                print(f"Epoch {epoch+1} 結果:")
                print(f"  訓練損失: {epoch_loss:.4f}, 訓練準確率: {epoch_acc:.2f}%")
                print(f"  測試損失: {test_loss:.4f}, 測試準確率: {test_acc:.2f}%")
                if len(self.mi_records['IXZ']) > 0:
                    print(f"  當前MI - IXZ: {self.mi_records['IXZ'][-1]:.4f}, IZY: {self.mi_records['IZY'][-1]:.4f}")
                
        except KeyboardInterrupt:
            print("\n訓練被中斷")
        finally:
            # 移除hook
            self.remove_hook()
        
        print("\n訓練完成!")
        print(f"總共處理了 {len(self.mi_records['IXZ'])} 個批次")
        if len(self.mi_records['IXZ']) > 0:
            print(f"最終MI值 - IXZ: {self.mi_records['IXZ'][-1]:.4f}, IZY: {self.mi_records['IZY'][-1]:.4f}")


def main():
    """主函數"""
    print("=== 簡單 CNN CIFAR10 互信息訓練測試 ===")
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 創建訓練器
    trainer = MITrainer(device=device, save_dir='./simple_cnn_mi_results')
    
    # 載入資料
    trainer.load_cifar10_data(batch_size=128, data_augmentation=False)
    
    # 創建模型
    trainer.create_model()
    
    # 開始訓練
    trainer.train(epochs=10, lr=0.001)  # 先訓練5個epochs進行測試
    
    # 保存結果
    trainer.save_results()
    
    # 繪製結果
    trainer.plot_results()
    
    print("\n=== 測試完成 ===")
    print("結果已保存到 ./simple_cnn_mi_results 目錄")


if __name__ == "__main__":
    main()