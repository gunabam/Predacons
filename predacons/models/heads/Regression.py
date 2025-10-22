from predacons.models import DataStructs
import torch
from torch import nn
from typing import List

class SentenceRegressionHead(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 target_dim: int = 768,
                 analyze_inputs: List[str] = ['a'],
                 **kwargs):
        super().__init__()
        self.training_task = 'sentence_regression'
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.regressor = nn.Linear(hidden_size, target_dim)
        self.loss_fct = nn.MSELoss()
        self.analyze_inputs = analyze_inputs
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        x = self.regressor(x)
        return x
    
    def regression(self,
                   pooled_output: torch.Tensor,
                   target: torch.Tensor,
                   is_target: torch.Tensor) -> DataStructs.ClassificationOutput:
        logits = self.forward(pooled_output)
        if is_target != None:
            logits = logits[is_target]
        loss = self.loss_fct(logits, target)
        return {'logits': logits, 'labels': target, 'loss': loss}