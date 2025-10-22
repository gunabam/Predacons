from transformers import T5EncoderModel as T5EncoderModelBase
from torch import nn
import torch

class T5Pooler(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class T5Model(nn.Module):

    def __init__(self, pretrained_dir: str, **kwargs):
        super().__init__()
        self.model = T5EncoderModelBase.from_pretrained(
            pretrained_dir,
            torch_dtype=torch.float16
        )
        self.pooler = T5Pooler(kwargs['hidden_size'])
    
    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.pooled_output = self.pooler(outputs.last_hidden_state)
        outputs.tokens = outputs.last_hidden_state
        return outputs