from predacons.configs.TransformerConfigs import TransformerConfig
from predacons.configs.HeadConfigs import HeadConfig
from predacons.architectures.forwards.MultiTaskForward import MultiTaskForward
from torch_geometric.data import Batch

from torch import nn
from torch.nn import ModuleDict
from typing import Optional, Dict, List, Literal

HeadName = str

class LanguageModelForMultiTask(MultiTaskForward):

    def __init__(self,
                 transformer_config: Optional[TransformerConfig] = None,
                 heads: Dict[HeadName, HeadConfig] = {},
                 inputs: List[Literal['a', 'b', 'c']] = ['a'],
                 initialize_weights: bool = True):
        super().__init__()
        self.inputs = inputs
        # model architecture
        self.transformer = transformer_config.get_model()
        self.heads = ModuleDict()
        for head_name, head_config in heads.items():
            self.heads[head_name] = head_config.get_model()
        # tensor names
        self.tensor_names = []
        for head_name, head in self.heads.items():
            for inp in head.analyze_inputs:
                if isinstance(inp, tuple):
                    inp = '___'.join(inp)
                for key in ['logits', 'labels']:
                    self.tensor_names.append((head_name, inp, key))
        # initialize weights
        if initialize_weights:
            self.apply(self._init_weights)
    
    def get_model_outputs(self, data: Batch) -> Batch:
        # clone data as it gets updated via the various steps
        data = data
        # transformer and pooler output
        outputs = self.transformer.get_model_outputs(
            input_ids=data.input_ids,
            attention_mask=data.attention_mask
        )
        data.pooled_output = outputs.pooled_output
        data.tokens = outputs.tokens
        return data

    def _init_weights(self, module: nn.Module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()