from predacons.models import DataStructs, helpers
import torch
from torch import nn
import numpy as np
from typing import List, Tuple
        
class SiameseSentenceConvergenceHead(nn.Module):

    def __init__(self,
                 margin: float = 1.0,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b', 'c')]):
        super().__init__()
        # input in order of (anchor, positive, negative)
        self.loss_fct = nn.TripletMarginLoss(margin=margin)
        self.training_task = 'siamese_sentence_convergence'
        self.analyze_inputs = analyze_inputs
    
    def forward(self):
        # currently no transformations are implimented
        # not necessary unless working with multimodule models
        pass
    
    def convergence(self,
                    pooled_output_a: torch.Tensor,
                    pooled_output_b: torch.Tensor,
                    pooled_output_c: torch.Tensor,
                    labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        # in multidataset training - the labels will identify which triplets are considered
        # ones to be ignored will have -100
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        pooled_output_a = helpers.recast_input_for_single_label(pooled_output_a, consider_index)
        pooled_output_b = helpers.recast_input_for_single_label(pooled_output_b, consider_index)
        pooled_output_c = helpers.recast_input_for_single_label(pooled_output_c, consider_index)
        loss = self.loss_fct(pooled_output_a, pooled_output_b, pooled_output_c)
        # the logits will have concatenation of anchor, positive, negative
        # torch.chunk will be used to recieve the tuple
        logits = torch.cat((pooled_output_a, pooled_output_b, pooled_output_c), 0)
        return {'logits': logits, 'labels': None, 'loss': loss}

class SiameseTokenConvergenceHead(nn.Module):

    def __init__(self,
                 margin: float = 1.0,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b', 'c')]):
        super().__init__()
        # input in order of (anchor, positive, negative)
        self.loss_fct = nn.TripletMarginLoss(margin=margin)
        self.training_task = 'siamese_token_convergence'
        self.analyze_inputs = analyze_inputs
    
    def forward(self):
        # currently no transformations are implimented
        # not necessary unless working with multimodule models
        pass
    
    def convergence(self,
                    tokens_a: torch.Tensor,
                    token_indexes_a: torch.Tensor,
                    tokens_b: torch.Tensor,
                    token_indexes_b: torch.Tensor,
                    tokens_c: torch.Tensor,
                    token_indexes_c: torch.Tensor,
                    labels: torch.Tensor) -> DataStructs.ClassificationOutput:
        # in multidataset training - the labels will identify which triplets are considered
        # ones to be ignored will have -100
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        token_indexes_a = helpers.recast_input_for_single_label(token_indexes_a, consider_index)
        token_indexes_b = helpers.recast_input_for_single_label(token_indexes_b, consider_index)
        token_indexes_c = helpers.recast_input_for_single_label(token_indexes_c, consider_index)
        tokens_a_trimmed = tokens_a.index_select(0, token_indexes_a)
        tokens_b_trimmed = tokens_b.index_select(0, token_indexes_b)
        tokens_c_trimmed = tokens_c.index_select(0, token_indexes_c)
        loss = self.loss_fct(tokens_a_trimmed, tokens_b_trimmed, tokens_c_trimmed)
        # the logits will have concatenation of anchor, positive, negative
        # torch.chunk will be used to recieve the tuple
        logits = torch.cat((tokens_a_trimmed, tokens_b_trimmed, tokens_c_trimmed), 0)
        return {'logits': logits, 'labels': None, 'loss': loss}