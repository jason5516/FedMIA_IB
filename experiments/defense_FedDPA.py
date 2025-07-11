import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd

class DefenseStrategy:
    """
    一個封裝了基於 Fisher Information 的個性化聯邦學習和差分隱私防禦機制的類別。

    這個類別提供了在客戶端進行本地訓練 (`local_update`) 和在服務端
    進行安全聚合 (`aggregate_updates`) 的方法。
    """
    def __init__(self, args):
        """
        初始化防禦策略。

        Args:
            args: 一個包含所有必要超參數的對象 (例如從 argparse 解析的結果)。
                  需要包含: lambda_1, lambda_2, clipping_bound, lr, 
                             local_epoch, fisher_threshold 等。
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_fisher_diag(self, model, dataloader):
        """
        計算模型在給定數據集上 Fisher Information Matrix 的對角線估計值。

        Args:
            model (torch.nn.Module): 要評估的模型。
            dataloader (torch.utils.data.DataLoader): 用於計算 Fisher 的數據加載器。

        Returns:
            list[torch.Tensor]: 一個列表，包含每個參數的正規化後 Fisher 對角線值。
        """
        model.eval()
        fisher_diag = [torch.zeros_like(param) for param in model.parameters()]
        model = model.to(self.device)

        for data, labels in dataloader:
            data, labels = data.to(self.device), labels.to(self.device)
            log_probs = torch.nn.functional.log_softmax(model(data), dim=1)

            for i, label in enumerate(labels):
                log_prob = log_probs[i, label]
                model.zero_grad()
                # retain_graph=True is needed for the loop over labels
                grad = autograd.grad(log_prob, model.parameters(), retain_graph=True)
                for fisher_val, grad_val in zip(fisher_diag, grad):
                    fisher_val.add_(grad_val.detach() ** 2)
        
        num_samples = len(dataloader.dataset)
        fisher_diag = [val / num_samples for val in fisher_diag]
        
        # 正規化 Fisher Information 到 [0, 1]
        normalized_fisher_diag = []
        for fisher_value in fisher_diag:
            x_min = torch.min(fisher_value)
            x_max = torch.max(fisher_value)
            # 處理 x_max == x_min 的邊界情況，避免除以零
            if x_max == x_min:
                normalized_fisher_value = torch.zeros_like(fisher_value)
            else:
                normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
            normalized_fisher_diag.append(normalized_fisher_value)

        model.to('cpu')
        return normalized_fisher_diag

    def _custom_loss(self, outputs, labels, param_diffs, reg_type):
        """內部使用的自定義損失函式，包含正規化項 R1 和 R2。"""
        ce_loss = F.cross_entropy(outputs, labels)
        if reg_type == "R1":
            reg_loss = (self.args.lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
        elif reg_type == "R2":
            C = self.args.clipping_bound
            norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            reg_loss = (self.args.lambda_2 / 2) * torch.norm(norm_diff - C)
        else:
            raise ValueError("無效的正規化類型 (Invalid regularization type)")
        return ce_loss + reg_loss

    def local_update(self, model, dataloader, global_model):
        """
        在客戶端執行的本地訓練過程。

        Args:
            model (torch.nn.Module): 客戶端本地模型。
            dataloader (torch.utils.data.DataLoader): 客戶端的本地數據加載器。
            global_model (torch.nn.Module): 當前的全局模型。

        Returns:
            list[torch.Tensor]: 計算出的模型更新 (本地模型與全局模型的參數差異)。
        """
        fisher_threshold = self.args.fisher_threshold
        model = model.to(self.device)
        global_model = global_model.to(self.device)

        w_glob = [param.clone().detach() for param in global_model.parameters()]

        fisher_diag = self.compute_fisher_diag(model, dataloader)

        # 根據 Fisher Information 劃分參數
        u_loc, v_loc = [], []
        for param, fisher_value in zip(model.parameters(), fisher_diag):
            u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
            u_loc.append(u_param)
            v_loc.append(v_param)

        u_glob, v_glob = [], []
        for global_param, fisher_value in zip(global_model.parameters(), fisher_diag):
            u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
            u_glob.append(u_param)
            v_glob.append(v_param)

        # 初始化模型：保留本地重要參數，繼承全局不重要參數
        for u_param, v_param, model_param in zip(u_loc, v_glob, model.parameters()):
            model_param.data = u_param + v_param

        # 階段一：訓練重要參數
        optimizer1 = optim.Adam(model.parameters(), lr=self.args.lr)
        for _ in range(self.args.local_epoch):
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer1.zero_grad()
                outputs = model(data)
                param_diffs = [p_new - p_old for p_new, p_old in zip(model.parameters(), w_glob)]
                loss = self._custom_loss(outputs, labels, param_diffs, "R1")
                loss.backward()
                with torch.no_grad():
                    for model_param, u_param in zip(model.parameters(), u_loc):
                        model_param.grad *= (u_param != 0)
                optimizer1.step()

        # 階段二：訓練不重要參數
        optimizer2 = optim.Adam(model.parameters(), lr=self.args.lr)
        for _ in range(self.args.local_epoch):
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer2.zero_grad()
                outputs = model(data)
                param_diffs = [p_new - p_old for p_new, p_old in zip(model.parameters(), w_glob)]
                loss = self._custom_loss(outputs, labels, param_diffs, "R2")
                loss.backward()
                with torch.no_grad():
                    for model_param, v_param in zip(model.parameters(), v_glob):
                        model_param.grad *= (v_param != 0)
                optimizer2.step()
        
        with torch.no_grad():
            update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]

        model.to('cpu')
        return update

    def aggregate_updates(self, global_model, client_updates, client_weights, noise_multiplier):
        """
        在服務端執行的差分隱私安全聚合。

        Args:
            global_model (torch.nn.Module): 需要被更新的全局模型。
            client_updates (list[list[torch.Tensor]]): 來自所有客戶端的模型更新列表。
            client_weights (list[float]): 對應每個客戶端更新的權重。
            noise_multiplier (float): 用於差分隱私的噪聲乘數。

        Returns:
            torch.nn.Module: 更新後的全局模型。
        """
        # 裁剪客戶端更新
        clipped_updates = []
        for client_update in client_updates:
            norm = torch.sqrt(sum([torch.sum(param ** 2) for param in client_update]))
            # 裁剪率不能小於 1
            clip_rate = max(1.0, norm.item() / self.args.clipping_bound)
            clipped_update = [(param / clip_rate) for param in client_update]
            clipped_updates.append(clipped_update)

        # 添加高斯噪聲
        noisy_updates = []
        for clipped_update in clipped_updates:
            # 噪聲標準差與裁剪邊界和噪聲乘數成正比
            noise_stddev = self.args.clipping_bound * noise_multiplier
            noise = [torch.randn_like(param) * noise_stddev for param in clipped_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(clipped_update, noise)]
            noisy_updates.append(noisy_update)

        # 加權聚合
        aggregated_update = [
            torch.sum(
                torch.stack(
                    [
                        noisy_update[param_index] * client_weights[idx]
                        for idx, noisy_update in enumerate(noisy_updates)
                    ]
                ),
                dim=0,
            )
            for param_index in range(len(noisy_updates[0]))
        ]

        # 更新全局模型
        with torch.no_grad():
            for global_param, update in zip(global_model.parameters(), aggregated_update):
                global_param.add_(update)
        
        return global_model