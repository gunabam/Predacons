from torchmetrics import (
    R2Score
)
from torch.nn import ModuleDict

def get(name: str, num_outputs: int = 1):
    metrics = ModuleDict()
    metrics[f'{name}_r2score'] = R2Score(num_outputs=num_outputs)
    return metrics