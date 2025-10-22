from predacons.lightning.bases.MultiTaskSingleOptimizerBase import (
    MultiTaskSingleOptimizerBase
)
from predacons.architectures.LanguageModelForMultiTask import LanguageModelForMultiTask
from predacons.configs.TransformerConfigs import TransformerConfig
from predacons.configs.HeadConfigs import HeadConfig
from typing import Dict, Optional, List, Literal

HeadName = str

class LanguageModelForMultiTaskLightning(MultiTaskSingleOptimizerBase):

    def __init__(self,
                 transformer_config: Optional[TransformerConfig] = None,
                 heads: Dict[HeadName, HeadConfig] = {},
                 inputs: List[Literal['a', 'b', 'c']] = ['a'],
                 initialize_weights: bool = True,
                 **kwargs):
        model = LanguageModelForMultiTask(transformer_config=transformer_config,
                                          heads=heads,
                                          inputs=inputs,
                                          initialize_weights=initialize_weights)
        super().__init__(model=model, **kwargs)