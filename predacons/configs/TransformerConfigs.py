from predacons.configs.Config import ConfigTemplate
from predacons.models.transformer.Bert import BertModel
from predacons.models.transformer.T5 import T5Model
from torch import nn
from typing import Union

class BertConfig(ConfigTemplate):

    def __init__(self, pretrained_dir: str):
        super().__init__(base='Bert',
                         properties={'pretrained_dir': pretrained_dir})
    
    def get_model(self) -> nn.Module:
        return BertModel(pretrained_dir=self.properties['pretrained_dir'])

class T5Config(ConfigTemplate):

    def __init__(self,
                 pretrained_dir: str,
                 hidden_size: int = 1024):
        super().__init__(base='T5',
                         properties={'pretrained_dir': pretrained_dir,
                                     'hidden_size': hidden_size})
    
    def get_model(self) -> nn.Module:
        return T5Model(pretrained_dir=self.properties['pretrained_dir'],
                       **self.properties)

TransformerConfig = Union[BertConfig, T5Config]