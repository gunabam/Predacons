import torch
from torch.nn import ModuleDict
from torchmetrics import Metric
from typing import Tuple

class TripletAccuracy(Metric):
    def __init__(self, margin: float = 1.0, p: float = 2.0):
        super().__init__()
        self.margin = margin
        self.pdist = torch.nn.PairwiseDistance(p=p, keepdim=False)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: None):
        anchor, positive, negative = torch.chunk(preds, 3, dim=0)
        # triplet distances
        pos_dist = self.pdist(anchor, positive)
        neg_dist = self.pdist(anchor, negative)
        # calculate difference in distance
        dist_diff = neg_dist - pos_dist
        # check if distance meets margin
        dist_diff = dist_diff >= self.margin
        self.correct += len(dist_diff.nonzero())
        self.total += len(anchor)
    
    def compute(self):
        return self.correct / self.total

def get(name: str, margin: float = 1.0, p: float = 2.0):
    metrics = ModuleDict()
    metrics[name] = TripletAccuracy(margin=margin, p=p)
    return metrics