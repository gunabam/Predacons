import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Union, Tuple

class MultiInputData(Data):

    def __init__(self, sentences: Dict[str, Data] = {}, common_y: Optional[Data] = None):
        super().__init__()
        # example keys for GraphModelForMultiTask: "a", "b", "c" ...
        self.sentences = sentences
        self.common_y = common_y
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        cache = {}
        common_y = getattr(self, 'common_y', None)
        if isinstance(common_y, Data):
            for k, v in common_y.to_dict().items():
                cache[f'common___{k}'] = v
        for inp, g in self.sentences.items():
            for k, v in g.to_dict().items():
                cache[f'batch___{k}___{inp}'] = v
        return cache

def batch_from_data_list(l: List[MultiInputData]) -> MultiInputData:
    keys = l[0].sentences.keys()
    sentences = {k: Batch.from_data_list([d.sentences[k] for d in l]) for k in keys}
    if getattr(l[0], 'common_y', None) == None:
        common_y = None
    else:
        common_y = Batch.from_data_list([d.common_y for d in l])
    return MultiInputData(sentences=sentences, common_y=common_y)

def recover_data(inputs: List[str], **kwargs) -> Dict[str, Data]:
    # parse and organize tensors by input
    batch = {inp: {} for inp in inputs}
    for x, y in kwargs.items():
        prefix = x.split('___')[0]
        suffix = x.split('___')[-1]
        interm = '___'.join(x.split('___')[1:-1])
        if prefix == 'batch' and suffix in batch:
            batch[suffix][interm] = y
    # cast as data objects
    out = {}
    for inp in inputs:
        out[inp] = Data(input_ids=batch[inp]['input_ids'],
                        attention_mask=batch[inp]['attention_mask'])
        for x, y in batch[inp].items():
            if x not in ['input_ids', 'attention_mask']:
                setattr(out[inp], x, y)
    return out