from predacons.data.DataClass import MultiInputData
from torch_geometric.data import Data, InMemoryDataset, Dataset
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
import math
import os
from typing import Dict, List, Optional, Union, Literal

DatasetName = str

class LanguageDataset(InMemoryDataset):

    def __init__(self,
                 csv_table_fp: Union[str, List[str]],
                 cache_fp: Optional[str] = None,
                 var_to_vocab_dict: Dict[str, Dict[str, int]] = {},
                 var_to_class_dict: Dict[str, Dict[str, int]] = {},
                 include_columns: List[str] = [],
                 add_cls_token: bool = True,
                 add_sep_token: bool = True,
                 subset: Optional[int] = None,
                 inputs: List[Literal['a', 'b', 'c']] = ['a'],
                 cls_token: str = '[CLS]',
                 sep_token: str = '[SEP]'):
        self.csv_table_fp = csv_table_fp
        self.cache_fp = cache_fp
        self.var_to_vocab_dict = var_to_vocab_dict
        self.var_to_class_dict = var_to_class_dict
        self.include_columns = include_columns
        self.add_cls_token = add_cls_token
        self.add_sep_token = add_sep_token
        self.subset = subset
        self.inputs = inputs
        self.cls_token = cls_token
        self.sep_token = sep_token
        super().__init__(root='',
                         transform=None,
                         pre_transform=None,
                         pre_filter=None)
        if self.cache_fp == None or os.path.exists(self.cache_fp) == False:
            self._process()
            if cache_fp != None:
                self.save()
        else:
            self.load()
    
    @property
    def processed_file_names(self):
        return ['']

    def _process(self):
        # "input" must be name for sentences
        # columns must be {name}___{input_suffix}
        # multiple input suffix is seperated by ___
        unk_id = self.var_to_vocab_dict['input']['[UNK]']
        data_list = []
        if isinstance(self.csv_table_fp, str):
            data = pd.read_csv(self.csv_table_fp).to_dict('records')
        else:
            data = pd.concat([pd.read_csv(fp) for fp in self.csv_table_fp]).to_dict('records')
        if isinstance(self.subset, int):
            data = data[:self.subset]
        for rec in tqdm(data):
            # create base sentences
            sentences = {}
            common_y = Data()
            for inp in self.inputs:
                input_ids = rec[f'input___{inp}'].split()
                if self.add_cls_token:
                    input_ids = [self.cls_token] + input_ids
                if self.add_sep_token:
                    input_ids = input_ids + [self.sep_token]
                input_ids = [self.var_to_vocab_dict['input'].get(tok, unk_id) for tok in input_ids]
                attention_mask = [1] * len(input_ids)
                sentences[inp] = Data(
                    input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask)
                )
            # add properties
            for col, value in rec.items():
                var = col.split('___')[0]
                if var == 'input': continue
                if var not in self.include_columns: continue
                num_inputs = len(col.split('___')) - 1
                if num_inputs == 0: continue
                elif num_inputs > 1:
                    if var in self.var_to_class_dict:
                        setattr(common_y, col, torch.LongTensor([self.var_to_class_dict[var].get(value, -100)]))
                    else:
                        setattr(common_y, col, torch.Tensor([[value]]))
                else:
                    inp = col.split('___')[-1]
                    if var in self.var_to_vocab_dict:
                        toks = [self.var_to_vocab_dict[var].get(tok, -100) for tok in value.split()]
                        if self.add_cls_token:
                            toks = [-100] + toks
                        if self.add_sep_token:
                            toks = toks + [-100]
                        setattr(sentences[inp], var, torch.LongTensor(toks))
                    elif var in self.var_to_class_dict:
                        setattr(sentences[inp], var, torch.LongTensor([self.var_to_class_dict[var].get(value, -100)]))
                    else:
                        setattr(sentences[inp], var, torch.Tensor([[value]]))
            data_list.append(MultiInputData(sentences=sentences, common_y=common_y))
        self._data, self.slices = self.collate(data_list)
    
    def save(self):
        torch.save((self._data, self.slices), self.cache_fp)
    
    def load(self):
        self._data, self.slices = torch.load(self.cache_fp)

class MultiLanguageDataset(InMemoryDataset):

    def __init__(self,
                 datasets: Dict[DatasetName, LanguageDataset] = {},
                 cache_fp: Optional[str] = None,
                 input_map: Dict[DatasetName, Dict[str, str]] = {}):
        self.cache_fp = cache_fp
        super().__init__(root='',
                         transform=None,
                         pre_transform=None,
                         pre_filter=None)
        if self.cache_fp == None or os.path.exists(self.cache_fp) == False:
            self._process(datasets, input_map)
            if cache_fp != None:
                self.save()
        else:
            self.load()

    @property
    def processed_file_names(self):
        return ['']
    
    def _process(self, datasets: Dict[DatasetName, LanguageDataset], input_map: Dict[DatasetName, Dict[str, str]]):
        # find largest dataset
        max_n = max([d.len() for d in datasets.values()])
        # dataset to indexes
        dataset_indexes = {name: list(range(d.len())) for name, d in datasets.items()}
        for name, indexes in tqdm(dataset_indexes.items()):
            repeats = math.floor(max_n / len(indexes))
            remainder = max_n % len(indexes)
            dataset_indexes[name] = indexes * repeats + indexes[:remainder]
        # create new data objects
        data_list = []
        keys = list(dataset_indexes.keys())
        values = list(dataset_indexes.values())
        zipped = list(zip(*values))
        for dp in tqdm(zipped):
            sentences = {}
            common_y = Data()
            for name, idx in zip(keys, dp):
                d = datasets[name].get(idx)
                # add to sentences
                for old_inp, s in d.sentences.items():
                    if old_inp not in input_map[name]: continue
                    new_inp = input_map[name][old_inp]
                    sentences[new_inp] = s
                # add to common y
                for k, v in d.common_y.to_dict().items():
                    old_inps = k.split('___')[1:]
                    new_inps = [input_map[name][inp] for inp in old_inps if inp in input_map[name]]
                    if len(new_inps) != len(old_inps): continue
                    new_k = k.split('___')[0] + '___' + '___'.join(new_inps)
                    setattr(common_y, new_k, v)
            data_list.append(MultiInputData(sentences=sentences, common_y=common_y))
        # collate
        self._data, self.slices = self.collate(data_list)

    def save(self):
        torch.save((self._data, self.slices), self.cache_fp)
    
    def load(self):
        self._data, self.slices = torch.load(self.cache_fp)

class MultiLanguageOffloadDataset(Dataset):

    def __init__(self,
                 datasets: Dict[DatasetName, LanguageDataset] = {},
                 cache_fp: Optional[str] = None,
                 input_map: Dict[DatasetName, Dict[str, str]] = {}):
        self.lookup = {}
        self.cache_fp = cache_fp
        super().__init__(root='',
                         transform=None,
                         pre_transform=None,
                         pre_filter=None)
        if len(self.processed_file_names) == 0:
            self._process(datasets=datasets, input_map=input_map)
        self.lookup = {idx: fp for idx, fp in enumerate(self.processed_file_names)}

    @property
    def processed_file_names(self):
        return glob(f'{self.cache_fp}/*')

    def len(self):
        return len(self.lookup)

    def get(self, idx):
        return torch.load(self.lookup[idx])
    
    def _process(self, datasets: Dict[DatasetName, LanguageDataset], input_map: Dict[DatasetName, Dict[str, str]]):
        # find largest dataset
        max_n = max([d.len() for d in datasets.values()])
        # dataset to indexes
        dataset_indexes = {name: list(range(d.len())) for name, d in datasets.items()}
        for name, indexes in tqdm(dataset_indexes.items()):
            repeats = math.floor(max_n / len(indexes))
            remainder = max_n % len(indexes)
            dataset_indexes[name] = indexes * repeats + indexes[:remainder]
        # create new data objects
        keys = list(dataset_indexes.keys())
        values = list(dataset_indexes.values())
        zipped = list(zip(*values))
        for dp_idx, dp in tqdm(enumerate(zipped), total=len(zipped)):
            if os.path.exists(f'{self.cache_fp}/{dp_idx}.pt'): continue
            sentences = {}
            common_y = Data()
            for name, idx in zip(keys, dp):
                d = datasets[name].get(idx)
                # add to sentences
                for old_inp, s in d.sentences.items():
                    if old_inp not in input_map[name]: continue
                    new_inp = input_map[name][old_inp]
                    sentences[new_inp] = s
                # add to common y
                for k, v in d.common_y.to_dict().items():
                    old_inps = k.split('___')[1:]
                    new_inps = [input_map[name][inp] for inp in old_inps if inp in input_map[name]]
                    if len(new_inps) != len(old_inps): continue
                    new_k = k.split('___')[0] + '___' + '___'.join(new_inps)
                    setattr(common_y, new_k, v)
            data = MultiInputData(sentences=sentences, common_y=common_y)
            torch.save(data, f'{self.cache_fp}/{dp_idx}.pt')