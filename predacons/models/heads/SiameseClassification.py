from predacons.models import DataStructs, helpers
import torch
from torch import nn
import numpy as np
from typing import List, Optional, Tuple

class SiameseSentenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size_a: int = 768,
                 hidden_size_b: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b')],
                 **kwargs):
        super().__init__()
        self.training_task = 'siamese_sentence_classification'
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size_a + hidden_size_b, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.analyze_inputs = analyze_inputs

    def forward(self,
                pooled_output_a: torch.Tensor,
                pooled_output_b: torch.Tensor) -> torch.Tensor:
        # concatenate the class nodes
        pooled_output = torch.cat((pooled_output_a, pooled_output_b), -1)
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return x

    def classification(self,
                       pooled_output_a: torch.Tensor,
                       pooled_output_b: torch.Tensor,
                       labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        pooled_output_a = helpers.recast_input_for_single_label(pooled_output_a, consider_index)
        pooled_output_b = helpers.recast_input_for_single_label(pooled_output_b, consider_index)
        logits = self.forward(pooled_output_a, pooled_output_b)
        loss = self.loss_fct(logits, labels)
        return {'logits': logits, 'labels': labels, 'loss': loss}

class SiameseTokenClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size_a: int = 768,
                 hidden_size_b: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b')],
                 **kwargs):
        super().__init__()
        self.training_task = 'siamese_token_classification'
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size_a + hidden_size_b, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.analyze_inputs = analyze_inputs

    def forward(self,
                token_a: torch.Tensor,
                token_indexes_a: torch.Tensor,
                token_b: torch.Tensor,
                token_indexes_b: torch.Tensor) -> torch.Tensor:
        # concatenate the class nodes
        token_a_trimmed = token_a.index_select(0, token_indexes_a)
        token_b_trimmed = token_b.index_select(0, token_indexes_b)
        x = torch.cat((token_a_trimmed, token_b_trimmed), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def classification(self,
                       token_a: torch.Tensor,
                       token_indexes_a: torch.Tensor,
                       token_b: torch.Tensor,
                       token_indexes_b: torch.Tensor,
                       labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        token_indexes_a = helpers.recast_input_for_single_label(token_indexes_a, consider_index)
        token_indexes_b = helpers.recast_input_for_single_label(token_indexes_b, consider_index)
        logits = self.forward(token_a, token_indexes_a, token_b, token_indexes_b)
        loss = self.loss_fct(logits, labels)
        return {'logits': logits, 'labels': labels, 'loss': loss}