from predacons.models.heads.TokenClassification import (
    SingleLabelTokenClassificationHead,
    MultiLabelTokenClassificationHead
)
from predacons.models.heads.SentenceClassification import (
    SingleLabelSentenceClassification,
    MultiLabelSentenceClassification
)
from predacons.models.heads.SiameseClassification import (
    SiameseSentenceClassificationHead,
    SiameseTokenClassificationHead
)
from predacons.models.heads.SiameseConvergence import (
    SiameseSentenceConvergenceHead,
    SiameseTokenConvergenceHead
)
from predacons.models.heads.Regression import (
    SentenceRegressionHead
)
from predacons.configs.Config import ConfigTemplate
from torch import nn
from typing import Union, List, Tuple, Optional

class TokenClsTaskHeadConfig(ConfigTemplate):
    
    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 multi_label: bool = False,
                 analyze_inputs: List[str] = ['a']):
        super().__init__(base='TokenClsTaskHead',
                         properties={'hidden_size': hidden_size,
                                     'hidden_dropout_prob': hidden_dropout_prob,
                                     'num_labels': num_labels,
                                     'class_weight': class_weight,
                                     'multi_label': multi_label,
                                     'analyze_inputs': analyze_inputs})
    
    def get_model(self) -> nn.Module:
        if self.properties['multi_label'] == True:
            return MultiLabelTokenClassificationHead(**self.properties)
        else:
            return SingleLabelTokenClassificationHead(**self.properties)

class SentenceClsTaskHeadConfig(ConfigTemplate):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 multi_label: bool = False,
                 analyze_inputs: List[str] = ['a']):
        super().__init__(base='SentenceClsTaskHead',
                         properties={'hidden_size': hidden_size,
                                     'hidden_dropout_prob': hidden_dropout_prob,
                                     'num_labels': num_labels,
                                     'class_weight': class_weight,
                                     'multi_label': multi_label,
                                     'analyze_inputs': analyze_inputs})

    def get_model(self) -> nn.Module:
        if self.properties['multi_label'] == True:
            return MultiLabelSentenceClassification(**self.properties)
        else:
            return SingleLabelSentenceClassification(**self.properties)

class SiameseSentenceClsTaskHeadConfig(ConfigTemplate):

    def __init__(self,
                 hidden_size_a: int = 768,
                 hidden_size_b: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b')]):
        super().__init__(base='SiameseSentenceClsTaskHead',
                         properties={'hidden_size_a': hidden_size_a,
                                     'hidden_size_b': hidden_size_b,
                                     'hidden_dropout_prob': hidden_dropout_prob,
                                     'num_labels': num_labels,
                                     'class_weight': class_weight,
                                     'analyze_inputs': analyze_inputs})
    
    def get_model(self) -> nn.Module:
        return SiameseSentenceClassificationHead(**self.properties)

class SiameseTokenClsTaskHeadConfig(ConfigTemplate):

    def __init__(self,
                 hidden_size_a: int = 768,
                 hidden_size_b: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 num_labels: int = 2,
                 class_weight: Optional[List[float]] = None,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b')]):
        super().__init__(base='SiameseTokenClsTaskHead',
                         properties={'hidden_size_a': hidden_size_a,
                                     'hidden_size_b': hidden_size_b,
                                     'hidden_dropout_prob': hidden_dropout_prob,
                                     'num_labels': num_labels,
                                     'class_weight': class_weight,
                                     'analyze_inputs': analyze_inputs})
    
    def get_model(self) -> nn.Module:
        return SiameseTokenClassificationHead(**self.properties)

class SiameseSentenceConvTaskHeadConfig(ConfigTemplate):

    def __init__(self,
                 margin: float = 1.0,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b', 'c')]):
        super().__init__(base='SiameseSentenceConvTaskHead',
                         properties={'margin': margin,
                                     'analyze_inputs': analyze_inputs})
    
    def get_model(self) -> nn.Module:
        return SiameseSentenceConvergenceHead(**self.properties)

class SiameseTokenConvTaskHeadConfig(ConfigTemplate):

    def __init__(self,
                 margin: float = 1.0,
                 analyze_inputs: List[Tuple[str, str]] = [('a', 'b', 'c')]):
        super().__init__(base='SiameseTokenConvTaskHead',
                         properties={'margin': margin,
                                     'analyze_inputs': analyze_inputs})
    
    def get_model(self) -> nn.Module:
        return SiameseTokenConvergenceHead(**self.properties)

class SentenceRegTaskHeadConfig(ConfigTemplate):

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_dropout_prob: float = 0.1,
                 target_dim: int = 768,
                 analyze_inputs: List[str] = ['a']):
        super().__init__(base='SentenceRegTaskHead',
                         properties={'hidden_size': hidden_size,
                                     'hidden_dropout_prob': hidden_dropout_prob,
                                     'target_dim': target_dim,
                                     'analyze_inputs': analyze_inputs})
    
    def get_model(self) -> nn.Module:
        return SentenceRegressionHead(**self.properties)

HeadConfig = Union[TokenClsTaskHeadConfig,
                   SentenceClsTaskHeadConfig,
                   SiameseSentenceClsTaskHeadConfig,
                   SiameseTokenClsTaskHeadConfig,
                   SiameseSentenceConvTaskHeadConfig,
                   SiameseTokenConvTaskHeadConfig,
                   SentenceRegTaskHeadConfig]