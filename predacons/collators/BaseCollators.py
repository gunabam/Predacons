from predacons.data.DataClass import MultiInputData, batch_from_data_list
from torch_geometric.data import Data
from dataclasses import dataclass
import torch
import copy
from typing import List, Dict, Tuple, Union

@dataclass
class BaseCollator:
    apply_batch: bool = False
    pad_tok_id: int = 1
    pad_variable_map: Dict[str, int] = None
    
    def __call__(self, data_list: List[MultiInputData]) -> Union[Dict[str, torch.Tensor], List[MultiInputData]]:
        data_list = copy.deepcopy(data_list)
        # preprocess
        for d in data_list:
            for k in d.sentences:
                d.sentences[k] = self.prepare_individual_data(d.sentences[k])
        # batch data
        if self.apply_batch:
            data_list = self.pad(data_list)
            data_list = self.adjust_dimensions(data_list)
            out = batch_from_data_list(data_list)
            # postprocess
            return self.postprocess(out)
        else:
            return data_list
    
    def pad(self, data_list: List[MultiInputData]) -> List[MultiInputData]:
        graph_keys = data_list[0].sentences.keys()
        for k in graph_keys:
            max_len = max([d.sentences[k].input_ids.shape[0] for d in data_list])
            for d in data_list:
                num_pads = max_len - d.sentences[k].input_ids.shape[0]
                if num_pads > 0:
                    # adjust input ids
                    d.sentences[k].input_ids = torch.cat([
                        d.sentences[k].input_ids,
                        torch.LongTensor([self.pad_tok_id] * num_pads)
                    ])
                    # adjust attention mask
                    d.sentences[k].attention_mask = torch.cat([
                        d.sentences[k].attention_mask,
                        torch.zeros(num_pads, dtype=torch.long)
                    ])
                    # adjust other variables
                    if isinstance(self.pad_variable_map, dict):
                        for var_name, var_pad_tok_id in self.pad_variable_map.items():
                            if hasattr(d.sentences[k], var_name):
                                new_var = torch.cat([
                                    getattr(d.sentences[k], var_name),
                                    torch.LongTensor([var_pad_tok_id] * num_pads)
                                ])
                                setattr(d.sentences[k], var_name, new_var)
        return data_list
    
    def adjust_dimensions(self, data_list: List[MultiInputData]) -> List[MultiInputData]:
        graph_keys = data_list[0].sentences.keys()
        for k in graph_keys:
            for d in data_list:
                d.sentences[k].input_ids = d.sentences[k].input_ids.reshape(1, -1)
                d.sentences[k].attention_mask = d.sentences[k].attention_mask.reshape(1, -1)
                if isinstance(self.pad_variable_map, dict):
                    for var_name in self.pad_variable_map:
                        if hasattr(d.sentences[k], var_name):
                            new_var = getattr(d.sentences[k], var_name).reshape(1, -1)
                            setattr(d.sentences[k], var_name, new_var)
        return data_list
            
    def prepare_individual_data(self, data: Data) -> Data:
        # method is overwritten (includes self-supervised training tasks)
        return data
    
    def postprocess(self, data: MultiInputData) -> Dict[str, torch.Tensor]:
        return data.to_dict()