from transformers import BertModel as BertModelBase
from torch import nn 

class BertModel(nn.Module):

    def __init__(self, pretrained_dir: str):
        super().__init__()
        self.model = BertModelBase.from_pretrained(pretrained_dir)
    
    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.pooled_output = outputs.pooler_output
        outputs.tokens = outputs.last_hidden_state
        return outputs