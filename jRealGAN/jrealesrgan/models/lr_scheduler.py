from collections import Counter
from jittor.optim import LRScheduler

class MultiStepRestartLR(LRScheduler):
    def __init__(self, optimizer, milestones=[], gamma=0.1, last_epoch=-1, restarts=(0, ), restart_weights=(1, )):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        #TODO set last_epoch is not ready
        super().__init__(optimizer, last_epoch)
    
    def get_gamma(self):
        if (self.last_epoch in self.milestones):
            return self.gamma
        return 1.0

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma**self.milestones[self.last_epoch] for group in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        self.update_lr()
            
    def update_lr(self):
        gamma = self.get_gamma()
        if gamma != 1.0:
            self.optimizer.lr = self.optimizer.lr * gamma
            for i, param_group in enumerate(self.optimizer.param_groups):
                if param_group.get("lr") != None:
                    param_group["lr"] = param_group["lr"] * gamma

