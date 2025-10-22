from predacons.models import DataStructs, helpers
import torch
from torch import nn
import numpy as np
from typing import List, Optional

class SingleLabelTokenClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[str] = ['a'],
                 **kwargs):
        super().__init__()
        self.training_task = 'token_classification'
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.analyze_inputs = analyze_inputs

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.dropout(tokens)
        tokens = self.classifier(tokens)
        return tokens
    
    def classification(self,
                       tokens: torch.Tensor,
                       labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        tokens = helpers.recast_input_for_single_label(tokens, consider_index)
        logits = self.forward(tokens)
        loss = self.loss_fct(logits, labels)
        return {'logits': logits, 'labels': labels, 'loss': loss}

class MultiLabelTokenClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[str] = ['a'],
                 **kwargs):
        super().__init__()
        self.training_task = 'token_classification'
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.ignore_index = -100 # if the first element corresponds to -100, ignore in loss calc
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
        self.analyze_inputs = analyze_inputs

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.dropout(tokens)
        tokens = self.classifier(tokens)
        return tokens
    
    def classification(self,
                       tokens: torch.Tensor,
                       labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_multi_label(labels)
        tokens = helpers.recast_input_for_multi_label(tokens, consider_index)
        logits = self.forward(tokens)
        loss = self.loss_fct(logits, labels)
        return {'logits': logits, 'labels': labels, 'loss': loss}