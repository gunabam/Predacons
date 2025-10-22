from predacons.collators.BaseCollators import BaseCollator
from torch_geometric.data import Data
from dataclasses import dataclass
import torch
from typing import Dict

@dataclass
class WordMaskCollator(BaseCollator):
    mask_tok_id: int = 1
    p: int = 0.15
    mask_name: str = 'mask'
    apply_batch: bool = False

    def prepare_individual_data(self, data: Data) -> Data:
        data = data.clone()
        out = self.process(data.input_ids)
        for k, v in out.items():
            setattr(data, k, v)
        
    def process(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        l = input_ids.shape[0]
        # generate mask
        masked_nodes = int(l*self.p)
        mask = torch.cat([
            torch.ones(masked_nodes, dtype=torch.bool),
            torch.zeros(l - masked_nodes, dtype=torch.bool)
        ])
        mask = mask.index_select(0, torch.randperm(mask.shape[0]))
        # introduce mask nodes
        y = x.reshape(-1).clone()
        x[mask] = torch.tensor([self.mask_tok_id])
        y[~mask] = -100 # non-masked words are ignored
        return {'x': x, self.mask_name: y}