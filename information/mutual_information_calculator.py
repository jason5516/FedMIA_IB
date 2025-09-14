import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Tuple, Optional


class MutualInformationCalculator:
    """
    獨立的互信息計算器，基於 Information Theory Experiment 中的實現
    
    用於計算神經網路層與輸入/輸出間的互信息
    支援輸入：原始資料X、標籤Y、層輸出Z、當前epoch
    """
    
    def __init__(self, device: str = 'cuda', num_layers: int = None):
        """
        初始化互信息計算器
        
        Args:
            device: 計算設備 ('cuda' 或 'cpu')
            num_layers: 神經網路層數（用於初始化 sigma 儲存）
        """
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        
        # 初始化 sigma 儲存（如果知道層數的話）
        if num_layers:
            self.sigmas = th.zeros((num_layers + 1, 1000)).to(device)  # 假設最多 1000 個 epoch
        else:
            self.sigmas = None
    
    def dist_mat(self, x: th.Tensor) -> th.Tensor:
        """
        計算距離矩陣
        
        Args:
            x: 輸入張量 [batch_size, features] 或 [batch_size, channels, height, width]
            
        Returns:
            距離矩陣 [batch_size, batch_size]
        """
        try:
            x = th.from_numpy(x)
        except (TypeError, AttributeError):
            x = x
        
        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)
        
        dist = th.norm(x[:, None] - x, dim=2)
        return dist
    
    def entropy(self, *args) -> th.Tensor:
        """
        計算熵
        
        Args:
            *args: 核矩陣列表
            
        Returns:
            計算得到的熵值
        """
        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val
        
        k /= k.trace()
        eigv = th.linalg.eigh(k).eigenvalues.abs()
        
        return -(eigv * (eigv.log2())).sum()
    
    def kernel_loss(self, k_x: th.Tensor, k_y: th.Tensor, k_l: th.Tensor) -> th.Tensor:
        """
        計算核損失
        
        Args:
            k_x: 輸入核矩陣
            k_y: 輸出核矩陣  
            k_l: 層核矩陣
            
        Returns:
            核損失值
        """
        beta = 1.0
        
        L = th.norm(k_l)
        Y = th.norm(k_y) ** beta
        X = th.norm(k_x) ** (1 - beta)
        
        LY = th.trace(th.matmul(k_l, k_y)) ** beta
        LX = th.trace(th.matmul(k_l, k_x)) ** (1 - beta)
        
        return 2 * th.log2((LY * LX) / (L * Y * X))
    
    def kernel_mat(self, x: th.Tensor, k_x: th.Tensor, k_y: th.Tensor, 
                   sigma: Optional[th.Tensor] = None, epoch: Optional[int] = None, 
                   idx: Optional[int] = None) -> th.Tensor:
        """
        計算核矩陣
        
        Args:
            x: 輸入資料
            k_x: 輸入核矩陣
            k_y: 輸出核矩陣
            sigma: 核參數，如果為 None 則自動計算
            epoch: 當前 epoch
            idx: 層索引
            
        Returns:
            核矩陣
        """
        d = self.dist_mat(x)
        
        if sigma is None:
            # 自動選擇 sigma
            if epoch is not None and epoch > 20:
                sigma_vals = th.linspace(0.3, 10 * d.mean(), 100).to(self.device)
            else:
                sigma_vals = th.linspace(0.3, 10 * d.mean(), 300).to(self.device)
            
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_x, k_y, k_l))
            
            # 更新 sigma 儲存
            if self.sigmas is not None and idx is not None and epoch is not None:
                if epoch == 0:
                    self.sigmas[idx + 1, epoch] = sigma_vals[L.index(max(L))]
                else:
                    self.sigmas[idx + 1, epoch] = 0.9 * self.sigmas[idx + 1, epoch - 1] + \
                                                    0.1 * sigma_vals[L.index(max(L))]
                sigma = self.sigmas[idx + 1, epoch]
            else:
                sigma = sigma_vals[L.index(max(L))]
            
        
        return th.exp(-d ** 2 / (sigma ** 2))
    
    def one_hot(self, y: th.Tensor, use_gpu: bool = True) -> th.Tensor:
        """
        轉換為 one-hot 編碼
        
        Args:
            y: 標籤張量
            use_gpu: 是否使用 GPU
            
        Returns:
            one-hot 編碼張量
        """
        try:
            y = th.from_numpy(y)
        except (TypeError, AttributeError):
            pass
        
        y_1d = y
        if use_gpu and self.device == 'cuda':
            y_hot = th.zeros((y.size(0), th.max(y).int() + 1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int() + 1))
        
        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1
        
        return y_hot
    
    def compute_mutual_information(self, 
                                 x: th.Tensor, 
                                 y: th.Tensor, 
                                 layer_outputs: List[th.Tensor], 
                                 epoch: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        計算互信息
        
        Args:
            x: 原始輸入資料 [batch_size, features]
            y: 標籤 [batch_size]
            layer_outputs: 各層輸出的列表，從第一層到最後一層
            epoch: 當前 epoch
            
        Returns:
            Tuple[I(X;T), I(T;Y)] - 返回 I(X;T) 和 I(T;Y) 的張量
            形狀為 [num_layers, 2]，其中第二維 0 是 I(X;T)，1 是 I(T;Y)
        """
        # 確保所有張量都在正確的設備上
        if self.device == 'cuda':
            x = x.cuda()
            y = y.cuda()
            layer_outputs = [output.cuda() for output in layer_outputs]
        
        # 準備資料：反轉層輸出順序（從輸出層到輸入層）
        data = layer_outputs.copy()
        data.reverse()
        
        # 對最後一層（原本的輸出層）應用 softmax
        data[-1] = self.softmax(data[-1])
        
        # 插入輸入和目標
        data.insert(0, x)  # 在開頭插入原始輸入
        data.append(self.one_hot(y, use_gpu=(self.device == 'cuda')))  # 在末尾插入 one-hot 標籤
        
        # 計算核矩陣
        k_x = self.kernel_mat(data[0], th.tensor([]), th.tensor([]), sigma=th.tensor(8.0))
        k_y = self.kernel_mat(data[-1], th.tensor([]), th.tensor([]), sigma=th.tensor(0.1))
        
        k_list = [k_x]
        for idx_l, val in enumerate(data[1:-1]):
            # 將層輸出 reshape 為 2D
            val_reshaped = val.reshape(data[0].size(0), -1)
            k_list.append(self.kernel_mat(val_reshaped, k_x, k_y,
                                        epoch=epoch, idx=idx_l))
            if self.sigmas is not None:
                print(f"Layer {idx_l}, Sigma: {self.sigmas[idx_l + 1, epoch]}")
        k_list.append(k_y)
        
        # 計算熵
        e_list = [self.entropy(i) for i in k_list]
        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]
        
        # 計算互信息
        num_layers = len(e_list) - 2  # 排除輸入和輸出
        MI = th.zeros((num_layers, 2))
        
        for idx_mi, val_mi in enumerate(e_list[1:-1]):
            MI[idx_mi, 0] = e_list[0] + val_mi - j_XT[idx_mi]  # I(X;T)
            MI[idx_mi, 1] = e_list[-1] + val_mi - j_TY[idx_mi]  # I(T;Y)
        
        return MI[:, 0], MI[:, 1]  # 返回 I(X;T), I(T;Y)
    
    def compute_mi_single_call(self, 
                              x: th.Tensor, 
                              y: th.Tensor, 
                              model: nn.Module, 
                              epoch: int,
                              extract_layers: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        """
        一次性計算互信息（包含模型前向傳播）
        
        Args:
            x: 原始輸入資料
            y: 標籤
            model: PyTorch 模型
            epoch: 當前 epoch
            extract_layers: 是否自動提取層輸出
            
        Returns:
            Tuple[I(X;T), I(T;Y)]
        """
        model.eval()
        
        with th.no_grad():
            # 獲取模型輸出
            model_output = model(x)
            
            if isinstance(model_output, tuple):
                data = model_output[0]  # 獲取層輸出，忽略 KL loss
            else:
                data = model_output
            
            # 轉換為列表格式
            if isinstance(data, tuple):
                data = list(data)
            elif not isinstance(data, list):
                data = [data]
            
            return self.compute_mutual_information(x, y, data, epoch)


def example_usage():
    """
    使用範例
    """
    # 創建計算器
    mi_calculator = MutualInformationCalculator(device='cuda', num_layers=3)
    
    # 模擬資料
    batch_size = 128
    input_dim = 784  # 例如 MNIST
    num_classes = 10
    
    x = th.randn(batch_size, input_dim).cuda()
    y = th.randint(0, num_classes, (batch_size,)).cuda()
    
    # 模擬層輸出
    layer1_output = th.randn(batch_size, 256).cuda()
    layer2_output = th.randn(batch_size, 128).cuda()
    layer3_output = th.randn(batch_size, num_classes).cuda()
    
    layer_outputs = [layer1_output, layer2_output, layer3_output]
    
    # 計算互信息
    I_XT, I_TY = mi_calculator.compute_mutual_information(x, y, layer_outputs, epoch=0)
    
    print(f"I(X;T): {I_XT}")
    print(f"I(T;Y): {I_TY}")
    
    return I_XT, I_TY


if __name__ == "__main__":
    example_usage()