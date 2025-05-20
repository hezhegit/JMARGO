import torch

class CustomLRAdjuster:
    def __init__(self, optimizer, threshold1=0.03, threshold2=0.02, decline1=0.001, decline2=0.002, min_lr=1e-5):
        self.optimizer = optimizer
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.decline1 = decline1
        self.decline2 = decline2
        self.min_lr = min_lr
        self.last_val_loss = float('inf')

    def step(self, current_val_loss):
        """
        根据验证集损失调整学习率
        Args:
            current_val_loss (float): 当前验证集上的损失
        """
        loss_diff = abs(self.last_val_loss - current_val_loss)

        if loss_diff >= self.threshold1:
            # 当损失变化大于等于 threshold1 时，不调整学习率
            pass
        elif self.threshold2 <= loss_diff < self.threshold1:
            # 当损失变化在 threshold2 和 threshold1 之间时，减少学习率 decline1
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr - self.decline1, self.min_lr)
                param_group['lr'] = new_lr
                print(
                    f'Learning rate adjusted from {old_lr} to {new_lr} due to loss change in range [{self.threshold2}, {self.threshold1})')
        elif loss_diff < self.threshold2:
            # 当损失变化小于 threshold2 时，减少学习率 decline2
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr - self.decline2, self.min_lr)
                param_group['lr'] = new_lr
                print(f'Learning rate adjusted from {old_lr} to {new_lr} due to loss change < {self.threshold2}')

        # 更新 last_val_loss
        self.last_val_loss = current_val_loss

    def state_dict(self):
        return {
            'threshold1': self.threshold1,
            'threshold2': self.threshold2,
            'decline1': self.decline1,
            'decline2': self.decline2,
            'last_val_loss': self.last_val_loss
        }

    def load_state_dict(self, state_dict):
        self.threshold1 = state_dict['threshold1']
        self.threshold2 = state_dict['threshold2']
        self.decline1 = state_dict['decline1']
        self.decline2 = state_dict['decline2']
        self.last_val_loss = state_dict['last_val_loss']
