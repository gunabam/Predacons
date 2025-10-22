from predacons.data.DataClass import MultiInputData
from predacons.collators.BaseCollators import BaseCollator
from torch_geometric.data import Data
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class MixedCollator:
    collators: Tuple[BaseCollator, ...] = ()
    standard_collator: BaseCollator = BaseCollator(apply_batch=True)

    def __call__(self, data_list: List[MultiInputData]) -> Dict[str, torch.Tensor]:
        # preprocess
        for d in data_list:
            for k in d.sentences:
                for c in self.collators:
                    d.sentences[k] = c.prepare_individual_data(d.sentences[k])
        return self.standard_collator(data_list)