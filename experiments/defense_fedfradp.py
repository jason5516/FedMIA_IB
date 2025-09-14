"""
FedFRADP Defense Implementation
基於 "Adaptive differential privacy with feedback regulation for robust model performance in federated learning" 論文實作

主要功能：
1. EMD-based 資料異質性量測
2. 適應性差分隱私機制
3. 回饋調節機制
4. 與現有 FL 系統的整合介面
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import warnings

class EMDHeterogeneityMeasure:
    """
    Earth Mover's Distance (EMD) 基於的資料異質性量測模組
    用於量化客戶端間的資料分佈差異
    """
    
    def __init__(self, feature_dim: int = 512, num_bins: int = 50):
        """
        初始化 EMD 異質性量測器
        
        Args:
            feature_dim: 特徵維度
            num_bins: 直方圖分箱數量
        """
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.client_distributions = {}
        
    def extract_features(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                        device: str = 'cuda') -> np.ndarray:
        """
        從模型中提取特徵用於分佈計算
        
        Args:
            model: 訓練好的模型
            data_loader: 資料載入器
            device: 計算設備
            
        Returns:
            提取的特徵陣列
        """
        model.eval()
        features = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(device)
                
                # 根據模型類型提取特徵
                if hasattr(model, 'get_features'):
                    # 對於 VAE 類型的模型
                    feature = model.get_features(data)
                elif hasattr(model, 'feature_extractor'):
                    # 對於有特徵提取器的模型
                    feature = model.feature_extractor(data)
                    feature = feature.view(feature.size(0), -1)
                else:
                    # 對於一般模型，使用倒數第二層
                    layers = list(model.children())
                    feature_extractor = nn.Sequential(*layers[:-1])
                    feature = feature_extractor(data)
                    feature = feature.view(feature.size(0), -1)
                
                features.append(feature.cpu().numpy())
                
                # 限制提取的批次數量以節省記憶體
                if batch_idx >= 10:
                    break
                    
        return np.concatenate(features, axis=0)
    
    def compute_distribution(self, features: np.ndarray) -> np.ndarray:
        """
        計算特徵的分佈直方圖
        
        Args:
            features: 特徵陣列
            
        Returns:
            正規化的分佈直方圖
        """
        # 對每個特徵維度計算直方圖
        distributions = []
        
        for dim in range(min(features.shape[1], self.feature_dim)):
            hist, _ = np.histogram(features[:, dim], bins=self.num_bins, density=True)
            distributions.append(hist)
            
        # 平均所有維度的分佈
        avg_distribution = np.mean(distributions, axis=0)
        
        # 正規化
        if np.sum(avg_distribution) > 0:
            avg_distribution = avg_distribution / np.sum(avg_distribution)
        else:
            avg_distribution = np.ones(self.num_bins) / self.num_bins
            
        return avg_distribution
    
    def register_client_distribution(self, client_id: str, model: nn.Module, 
                                   data_loader: torch.utils.data.DataLoader, 
                                   device: str = 'cuda'):
        """
        註冊客戶端的資料分佈
        
        Args:
            client_id: 客戶端 ID
            model: 客戶端模型
            data_loader: 客戶端資料載入器
            device: 計算設備
        """
        features = self.extract_features(model, data_loader, device)
        distribution = self.compute_distribution(features)
        self.client_distributions[client_id] = distribution
        
    def compute_emd_matrix(self) -> np.ndarray:
        """
        計算所有客戶端間的 EMD 距離矩陣
        
        Returns:
            EMD 距離矩陣
        """
        client_ids = list(self.client_distributions.keys())
        n_clients = len(client_ids)
        
        if n_clients < 2:
            return np.zeros((n_clients, n_clients))
            
        emd_matrix = np.zeros((n_clients, n_clients))
        
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i != j:
                    dist_i = self.client_distributions[client_i]
                    dist_j = self.client_distributions[client_j]
                    
                    # 使用簡化的 EMD 計算 (L1 距離的累積差異)
                    emd_distance = self._compute_simple_emd(dist_i, dist_j)
                    emd_matrix[i, j] = emd_distance
                    
        return emd_matrix
    
    def _compute_simple_emd(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        計算簡化的 EMD 距離 (使用累積分佈函數的 L1 距離)
        
        Args:
            dist1: 第一個分佈
            dist2: 第二個分佈
            
        Returns:
            簡化的 EMD 距離
        """
        # 計算累積分佈函數
        cdf1 = np.cumsum(dist1)
        cdf2 = np.cumsum(dist2)
        
        # 計算 CDF 之間的 L1 距離作為 EMD 的近似
        emd_distance = np.sum(np.abs(cdf1 - cdf2))
        
        return emd_distance
    
    def compute_heterogeneity_score(self, client_id: str) -> float:
        """
        計算特定客戶端的異質性分數
        
        Args:
            client_id: 客戶端 ID
            
        Returns:
            異質性分數 (0-1 之間，越高表示越異質)
        """
        if client_id not in self.client_distributions:
            return 0.5  # 預設中等異質性
            
        emd_matrix = self.compute_emd_matrix()
        client_ids = list(self.client_distributions.keys())
        
        if client_id not in client_ids:
            return 0.5
            
        client_idx = client_ids.index(client_id)
        
        # 計算該客戶端與其他所有客戶端的平均 EMD 距離
        if len(client_ids) <= 1:
            return 0.3  # 只有一個客戶端時返回中等異質性
            
        avg_emd = np.mean([emd_matrix[client_idx, j] for j in range(len(client_ids)) if j != client_idx])
        
        # 正規化到 0-1 範圍，並限制異質性分數的最大值
        max_possible_emd = 2.0  # 調整理論最大 EMD 距離
        heterogeneity_score = min(avg_emd / max_possible_emd, 0.8)  # 限制最大異質性分數為 0.8
        
        return heterogeneity_score


class AdaptiveDifferentialPrivacy:
    """
    適應性差分隱私機制
    根據資料異質性動態調整隱私預算和雜訊強度
    """
    
    def __init__(self, base_epsilon: float = 1.0, base_delta: float = 1e-5,
                 sensitivity: float = 1.0, heterogeneity_weight: float = 0.5):
        """
        初始化適應性差分隱私機制
        
        Args:
            base_epsilon: 基礎隱私預算
            base_delta: 基礎 delta 參數
            sensitivity: 敏感度參數
            heterogeneity_weight: 異質性權重 (0-1)
        """
        self.base_epsilon = base_epsilon
        self.base_delta = base_delta
        self.sensitivity = sensitivity
        self.heterogeneity_weight = heterogeneity_weight
        
    def compute_adaptive_epsilon(self, heterogeneity_score: float,
                               performance_feedback: float = 1.0) -> float:
        """
        根據異質性分數和效能回饋計算適應性隱私預算
        
        Args:
            heterogeneity_score: 異質性分數 (0-1)
            performance_feedback: 效能回饋分數 (0-1)
            
        Returns:
            適應性隱私預算
        """
        # 異質性越高，隱私預算越大 (允許更多雜訊)
        heterogeneity_factor = 1.0 + self.heterogeneity_weight * heterogeneity_score
        
        # 效能回饋越差，隱私預算越大 (減少雜訊強度)
        performance_factor = max(0.5, performance_feedback)  # 避免過小的效能因子
        
        adaptive_epsilon = self.base_epsilon * heterogeneity_factor * performance_factor
        
        # 確保隱私預算在合理範圍內，避免過小導致雜訊過大
        adaptive_epsilon = max(0.5, min(adaptive_epsilon, 5.0))
        
        return adaptive_epsilon
    
    def compute_noise_scale(self, adaptive_epsilon: float) -> float:
        """
        根據適應性隱私預算計算雜訊尺度
        
        Args:
            adaptive_epsilon: 適應性隱私預算
            
        Returns:
            高斯雜訊的標準差
        """
        # 根據差分隱私理論計算雜訊尺度
        noise_scale = (self.sensitivity * math.sqrt(2 * math.log(1.25 / self.base_delta))) / adaptive_epsilon
        # 限制雜訊尺度在合理範圍內，避免過大的雜訊
        max_noise_scale = 0.01  # 設定最大雜訊尺度
        noise_scale = min(noise_scale, max_noise_scale)
        return noise_scale
    
    def add_gaussian_noise(self, tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """
        為張量添加高斯雜訊
        
        Args:
            tensor: 輸入張量
            noise_scale: 雜訊尺度
            
        Returns:
            添加雜訊後的張量
        """
        if noise_scale <= 0:
            return tensor
        
        # 根據張量的標準差調整雜訊尺度，避免雜訊過大
        tensor_std = torch.std(tensor).item()
        if tensor_std > 0:
            # 限制雜訊不超過張量標準差的 10%
            adjusted_noise_scale = min(noise_scale, tensor_std * 0.1)
        else:
            adjusted_noise_scale = min(noise_scale, 0.001)
            
        noise = torch.normal(0, adjusted_noise_scale, size=tensor.shape, device=tensor.device)
        return tensor + noise
    
    def apply_adaptive_noise(self, model_params: Dict[str, torch.Tensor], 
                           heterogeneity_score: float,
                           performance_feedback: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        對模型參數應用適應性雜訊
        
        Args:
            model_params: 模型參數字典
            heterogeneity_score: 異質性分數
            performance_feedback: 效能回饋分數
            
        Returns:
            添加雜訊後的模型參數
        """
        adaptive_epsilon = self.compute_adaptive_epsilon(heterogeneity_score, performance_feedback)
        noise_scale = self.compute_noise_scale(adaptive_epsilon)
        
        noisy_params = {}
        for name, param in model_params.items():
            noisy_params[name] = self.add_gaussian_noise(param, noise_scale)
            
        return noisy_params


class FeedbackRegulator:
    """
    回饋調節機制
    根據全域模型效能動態調整隱私參數
    """
    
    def __init__(self, target_accuracy: float = 0.8, adjustment_rate: float = 0.1,
                 history_window: int = 5):
        """
        初始化回饋調節器
        
        Args:
            target_accuracy: 目標準確率
            adjustment_rate: 調整率
            history_window: 歷史視窗大小
        """
        self.target_accuracy = target_accuracy
        self.adjustment_rate = adjustment_rate
        self.history_window = history_window
        self.accuracy_history = []
        self.feedback_scores = []
        
    def update_performance(self, accuracy: float):
        """
        更新效能記錄
        
        Args:
            accuracy: 當前準確率
        """
        self.accuracy_history.append(accuracy)
        
        # 保持歷史視窗大小
        if len(self.accuracy_history) > self.history_window:
            self.accuracy_history.pop(0)
            
    def compute_feedback_score(self) -> float:
        """
        計算效能回饋分數
        
        Returns:
            回饋分數 (0.5-1.5)
        """
        if not self.accuracy_history:
            return 1.0
            
        current_accuracy = self.accuracy_history[-1]
        
        # 計算與目標準確率的差距
        accuracy_gap = current_accuracy - self.target_accuracy
        
        # 計算趨勢 (如果有足夠的歷史資料)
        trend = 0.0
        if len(self.accuracy_history) >= 3:
            recent_trend = np.mean(np.diff(self.accuracy_history[-3:]))
            trend = recent_trend
            
        # 綜合考慮當前效能和趨勢，使用更保守的調整
        feedback_score = 1.0 + self.adjustment_rate * accuracy_gap * 0.5 + 0.05 * trend
        
        # 限制在更窄的合理範圍內，避免過度調整
        feedback_score = max(0.5, min(feedback_score, 1.5))
        
        self.feedback_scores.append(feedback_score)
        return feedback_score
    
    def should_adjust_privacy(self) -> bool:
        """
        判斷是否需要調整隱私參數
        
        Returns:
            是否需要調整
        """
        if len(self.accuracy_history) < 2:
            return False
            
        # 如果準確率持續下降，需要調整
        recent_accuracies = self.accuracy_history[-3:]
        if len(recent_accuracies) >= 2:
            trend = np.mean(np.diff(recent_accuracies))
            return trend < -0.05  # 準確率下降超過 5%
            
        return False


class FedFRADPDefense:
    """
    FedFRADP 防禦機制主類別
    整合 EMD 異質性量測、適應性差分隱私和回饋調節
    結合 FedDPA 風格的服務端差分隱私聚合
    """
    
    def __init__(self, base_epsilon: float = 1.0, base_delta: float = 1e-5,
                 target_accuracy: float = 0.8, feature_dim: int = 512,
                 clipping_bound: float = 1.0, server_noise_multiplier: float = 0.1):
        """
        初始化 FedFRADP 防禦機制
        
        Args:
            base_epsilon: 基礎隱私預算
            base_delta: 基礎 delta 參數
            target_accuracy: 目標準確率
            feature_dim: 特徵維度
            clipping_bound: 服務端聚合時的裁剪邊界
            server_noise_multiplier: 服務端聚合時的噪聲乘數
        """
        self.emd_measure = EMDHeterogeneityMeasure(feature_dim=feature_dim)
        self.adaptive_dp = AdaptiveDifferentialPrivacy(
            base_epsilon=base_epsilon,
            base_delta=base_delta
        )
        self.feedback_regulator = FeedbackRegulator(target_accuracy=target_accuracy)
        
        # FedDPA 風格的服務端差分隱私參數
        self.clipping_bound = clipping_bound
        self.server_noise_multiplier = server_noise_multiplier
        
        # 記錄統計資訊
        self.round_stats = {
            'heterogeneity_scores': [],
            'adaptive_epsilons': [],
            'noise_scales': [],
            'feedback_scores': [],
            'accuracies': [],
            'clipping_norms': [],
            'server_noise_scales': []
        }
        
    def register_client(self, client_id: str, model: nn.Module, 
                       data_loader: torch.utils.data.DataLoader, device: str = 'cuda'):
        """
        註冊客戶端
        
        Args:
            client_id: 客戶端 ID
            model: 客戶端模型
            data_loader: 客戶端資料載入器
            device: 計算設備
        """
        self.emd_measure.register_client_distribution(client_id, model, data_loader, device)
        
    def apply_defense(self, client_id: str, model_params: Dict[str, torch.Tensor],
                     current_accuracy: float = None) -> Dict[str, torch.Tensor]:
        """
        對客戶端模型參數應用 FedFRADP 防禦
        
        Args:
            client_id: 客戶端 ID
            model_params: 模型參數
            current_accuracy: 當前準確率
            
        Returns:
            防禦後的模型參數
        """
        # 1. 計算異質性分數
        heterogeneity_score = self.emd_measure.compute_heterogeneity_score(client_id)
        
        # 2. 更新效能記錄並計算回饋分數
        feedback_score = 1.0
        if current_accuracy is not None:
            self.feedback_regulator.update_performance(current_accuracy)
            feedback_score = self.feedback_regulator.compute_feedback_score()
            
        # 3. 應用適應性差分隱私
        defended_params = self.adaptive_dp.apply_adaptive_noise(
            model_params, heterogeneity_score, feedback_score
        )
        
        # 4. 記錄統計資訊
        adaptive_epsilon = self.adaptive_dp.compute_adaptive_epsilon(
            heterogeneity_score, feedback_score
        )
        noise_scale = self.adaptive_dp.compute_noise_scale(adaptive_epsilon)
        
        self.round_stats['heterogeneity_scores'].append(heterogeneity_score)
        self.round_stats['adaptive_epsilons'].append(adaptive_epsilon)
        self.round_stats['noise_scales'].append(noise_scale)
        self.round_stats['feedback_scores'].append(feedback_score)
        if current_accuracy is not None:
            self.round_stats['accuracies'].append(current_accuracy)
            
        return defended_params
    
    def aggregate_with_dp(self, client_updates: List[Dict[str, torch.Tensor]],
                         client_weights: List[float],
                         adaptive_noise_multiplier: float = None) -> Dict[str, torch.Tensor]:
        """
        使用 FedDPA 風格的差分隱私進行服務端聚合
        
        Args:
            client_updates: 客戶端模型更新列表
            client_weights: 客戶端權重列表
            adaptive_noise_multiplier: 適應性噪聲乘數（如果為 None 則使用預設值）
            
        Returns:
            聚合後的模型參數
        """
        if not client_updates:
            raise ValueError("客戶端更新列表不能為空")
            
        # 使用適應性噪聲乘數或預設值
        noise_multiplier = adaptive_noise_multiplier if adaptive_noise_multiplier is not None else self.server_noise_multiplier
        
        # 1. 裁剪客戶端更新
        clipped_updates = []
        clipping_norms = []
        
        for client_update in client_updates:
            # 計算更新的 L2 範數
            update_norm = torch.sqrt(sum([torch.sum(param ** 2) for param in client_update.values()]))
            clipping_norms.append(update_norm.item())
            
            # 裁剪率不能小於 1
            clip_rate = max(1.0, update_norm.item() / self.clipping_bound)
            
            # 裁剪更新
            clipped_update = {}
            for name, param in client_update.items():
                clipped_update[name] = param / clip_rate
            clipped_updates.append(clipped_update)
        
        # 2. 添加高斯噪聲
        noisy_updates = []
        noise_stddev = self.clipping_bound * noise_multiplier
        
        for clipped_update in clipped_updates:
            noisy_update = {}
            for name, param in clipped_update.items():
                noise = torch.randn_like(param) * noise_stddev
                noisy_update[name] = param + noise
            noisy_updates.append(noisy_update)
        
        # 3. 加權聚合
        if not noisy_updates:
            raise ValueError("沒有有效的客戶端更新")
            
        # 初始化聚合結果
        aggregated_update = {}
        param_names = list(noisy_updates[0].keys())
        
        for name in param_names:
            # 加權平均
            weighted_sum = torch.zeros_like(noisy_updates[0][name])
            for i, noisy_update in enumerate(noisy_updates):
                weighted_sum += noisy_update[name] * client_weights[i]
            aggregated_update[name] = weighted_sum
        
        # 記錄統計資訊
        self.round_stats['clipping_norms'].extend(clipping_norms)
        self.round_stats['server_noise_scales'].append(noise_stddev)
        
        return aggregated_update
    
    def compute_adaptive_server_noise(self, heterogeneity_scores: List[float],
                                    feedback_score: float = 1.0) -> float:
        """
        根據異質性分數和回饋分數計算適應性服務端噪聲乘數
        
        Args:
            heterogeneity_scores: 所有參與客戶端的異質性分數列表
            feedback_score: 效能回饋分數
            
        Returns:
            適應性噪聲乘數
        """
        if not heterogeneity_scores:
            return self.server_noise_multiplier
            
        # 計算平均異質性分數
        avg_heterogeneity = np.mean(heterogeneity_scores)
        
        # 異質性越高，需要更多噪聲來保護隱私
        # 但效能回饋越差，需要減少噪聲來提升效能
        heterogeneity_factor = 1.0 + 0.5 * avg_heterogeneity
        performance_factor = feedback_score
        
        adaptive_noise_multiplier = self.server_noise_multiplier * heterogeneity_factor / performance_factor
        
        # 限制在合理範圍內
        adaptive_noise_multiplier = max(0.01, min(adaptive_noise_multiplier, 1.0))
        
        return adaptive_noise_multiplier
    
    def get_defense_stats(self) -> Dict:
        """
        獲取防禦統計資訊
        
        Returns:
            統計資訊字典
        """
        stats = {}
        for key, values in self.round_stats.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1] if values else 0
                }
        return stats
    
    def reset_stats(self):
        """重置統計資訊"""
        for key in self.round_stats:
            self.round_stats[key] = []
            
    def save_stats(self, filepath: str):
        """
        儲存統計資訊到檔案
        
        Args:
            filepath: 檔案路徑
        """
        import json
        stats = self.get_defense_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


# 便利函數和整合介面
def create_fedfradp_defense(config: Dict) -> FedFRADPDefense:
    """
    根據配置建立 FedFRADP 防禦機制
    
    Args:
        config: 配置字典
        
    Returns:
        FedFRADP 防禦實例
    """
    return FedFRADPDefense(
        base_epsilon=config.get('base_epsilon', 1.0),
        base_delta=config.get('base_delta', 1e-5),
        target_accuracy=config.get('target_accuracy', 0.8),
        feature_dim=config.get('feature_dim', 512),
        clipping_bound=config.get('clipping_bound', 1.0),
        server_noise_multiplier=config.get('server_noise_multiplier', 0.1)
    )


def integrate_with_fl_training(model: nn.Module, defense: FedFRADPDefense,
                              client_id: str, data_loader: torch.utils.data.DataLoader,
                              device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    與聯邦學習訓練流程整合的便利函數
    
    Args:
        model: 訓練後的模型
        defense: FedFRADP 防禦實例
        client_id: 客戶端 ID
        data_loader: 資料載入器
        device: 計算設備
        
    Returns:
        防禦後的模型參數
    """
    # 註冊客戶端 (如果尚未註冊)
    if client_id not in defense.emd_measure.client_distributions:
        defense.register_client(client_id, model, data_loader, device)
    
    # 獲取模型參數
    model_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # 應用防禦
    defended_params = defense.apply_defense(client_id, model_params)
    
    return defended_params


# 使用範例配置
DEFAULT_CONFIG = {
    'base_epsilon': 1.0,  # 降低基礎隱私預算，減少雜訊
    'base_delta': 1e-5,
    'target_accuracy': 0.6,
    'feature_dim': 512,
    'heterogeneity_weight': 0.3,  # 適度增加異質性權重
    'adjustment_rate': 0.1,
    'history_window': 5,
    'max_noise_scale': 0.01,  # 限制最大雜訊尺度
    'gradient_clip_norm': 1.0,
    'clipping_bound': 1.0,
    'server_noise_multiplier': 0.01  # 大幅降低服務端雜訊
}