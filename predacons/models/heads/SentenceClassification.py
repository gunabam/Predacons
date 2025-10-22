from predacons.models import DataStructs, helpers
import torch
from torch import nn
import numpy as np
from typing import Optional, List

class SingleLabelSentenceClassification(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[str] = ['a'],
                 **kwargs):
        super().__init__()
        self.training_task = 'sentence_classification'
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.analyze_inputs = analyze_inputs
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return x
    
    def classification(self,
                       pooled_output: torch.Tensor,
                       labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        consider_index, new_labels = helpers.recast_labels_for_single_label(labels)
        new_pooled_output = helpers.recast_input_for_single_label(pooled_output, consider_index)
        logits = self.forward(new_pooled_output)
        loss = self.loss_fct(logits, new_labels)
        if torch.isnan(loss):
            # place holder values (if all values in batch is -100)
            # to ensure that there are no wholes in metrics 
            new_labels = torch.LongTensor([0])
            logits = torch.Tensor([[1] + [0] * (self.num_labels - 1)])
            new_labels = new_labels.to(device=pooled_output.device)
            logits = logits.to(device=pooled_output.device)
            return {'logits': logits, 'labels': new_labels, 'loss': loss}
        else:
            return {'logits': logits, 'labels': new_labels, 'loss': loss}

class MultiLabelSentenceClassification(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[str] = ['a'],
                 **kwargs):
        super().__init__()
        self.training_task = 'multi_label_sentence_classification'
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

    def forward(self,
                pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return x
    
    def classification(self,
                       pooled_output: torch.Tensor,
                       labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_multi_label(labels)
        pooled_output = helpers.recast_input_for_multi_label(pooled_output, consider_index)
        logits = self.forward(pooled_output)
        loss = self.loss_fct(logits, labels)
        return {'logits': logits, 'labels': labels, 'loss': loss}