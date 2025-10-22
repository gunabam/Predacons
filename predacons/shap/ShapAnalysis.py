import shap
from shap._explanation import Explanation
from transformers import PreTrainedTokenizerFast
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from IPython.core.display import display, HTML
from typing import TypedDict, Dict, Optional, List, Literal, Union

class TokenizedInputDict(TypedDict):
     input_ids: np.array
     attention_mask: np.array
     token_type_ids: np.array

class SequenceClassificationShapAnalysis:
     
     def __init__(self,
                  base_fp: str,
                  head_fp: str,
                  tokenizer: PreTrainedTokenizerFast,
                  label2id: Dict[str, int], 
                  gpu_id: int):
          # NOTE base_fp and head_fp have to be onnx model filepaths
          providers = self.get_providers(gpu_id=gpu_id)
          self.model = self.get_model(model_fp=base_fp, providers=providers)
          self.tokenizer = tokenizer
          self.head = self.get_model(model_fp=head_fp, providers=providers)
          self.label2id = label2id
          self.labels = self.get_labels(label2id)
          self.text = None

    ################################################################################################
    # Load models
    ################################################################################################

     @staticmethod
     def get_providers(gpu_id: Optional[int]):
          if isinstance(gpu_id, int):
               providers = [('CUDAExecutionProvider', {'device_id': gpu_id})]
          else:
               providers = ['CPUExecutionProvider']
          return providers

     @staticmethod
     def get_model(model_fp: str, providers: List[str]):
          model = ort.InferenceSession(model_fp, providers=providers)
          return model
     
     @staticmethod
     def get_labels(label2id: Dict[str, int]):
          id2label = {y: x for x, y in label2id.items()}
          return [id2label[idx] for idx, _ in enumerate(id2label)]

     ################################################################################################
     # Core functions
     ################################################################################################

     def set_text(self, text: List[str], mask_region: int):
          self.text = text
          self.mask_region = mask_region

     def __call__(self, mask: np.array):
          out = []
          for m in tqdm(mask, leave=False):
               sequence = self.get_masked_sequece(m)
               tokenized_inputs = self.preprocess(sequence)
               out.append(self._forward(tokenized_inputs))
          return np.array(out)

     def get_masked_sequece(self, m: np.array):
          text = list(self.text)
          max_len = len(text)
          for idx, v in enumerate(m):
               if v == True:
                    if self.mask_region > 0:
                         min_idx = idx - self.mask_region
                         min_idx = 0 if min_idx < 0 else min_idx
                         max_idx = idx + self.mask_region
                         max_idx = max_len if max_idx > max_len else max_idx
                         text[min_idx:max_idx] = ['[MASK]'] * (max_idx - min_idx)
                    else:
                         text[idx] = '[MASK]'
          return ' '.join(text)
     
     def preprocess(self, sequence: str) -> TokenizedInputDict:
          return self.tokenizer(sequence, padding=True, return_tensors='np')

     @staticmethod
     def softmax(x):
          return np.exp(x) / np.exp(x).sum(-1, keepdims=True)
     
     def _forward(self, tokenized_inputs: TokenizedInputDict) -> np.array:
          pooled_output = self.model.run(['pooler_output'], dict(tokenized_inputs))[0]
          preds = self.head.run(['output'], {'input': pooled_output})[0][0]
          preds = self.softmax(preds)
          return preds
     
     @staticmethod
     def masker(mask: np.array, x: np.array):
          return mask.reshape((1, -1))

     def get_shap_values(self, text: List[str], max_evals: Union[str, int] = 'auto', mask_region: int = 0) -> Explanation:
          if isinstance(text, list) == False or isinstance(text[0], str) == False:
               return 'Text must be list of strings'
          else:
               x = np.ones((1, len(text)))
               self.set_text(text, mask_region)
               explainer = shap.Explainer(self, masker=self.masker)
               shap_values = explainer(x, max_evals=max_evals)
               return shap_values

     @staticmethod
     def highlighter(word: str, colour: Literal['red', 'blue'], opacity: float):
          if colour == 'red':
               r1, r2, r3 = 255, 0, 0
          else:
               r1, r2, r3 = 0, 0, 255
          word = f'<span style="background-color: rgba({r1}, {r2}, {r3}, {opacity})"> {word} </span>'
          return word

     def get_html(self, shap_values: Explanation, label: Optional[str] = None):
          if label == None:
               sequence = ' '.join(self.text)
               tokenized_inputs = self.preprocess(sequence)
               preds = self._forward(tokenized_inputs)
               selected_pred_idx = np.argmax(preds)
               score = round(preds[selected_pred_idx], 2)
               label = self.labels[selected_pred_idx]
               # find best prediction
               print(f'Best Prediction: {label}, score: {score}, label_idx: {selected_pred_idx}')
          else:
               selected_pred_idx = self.label2id[label]
          # find shapley values corresponding to prediction
          shapley_values = [round(v, 2) for v in shap_values[:, :, selected_pred_idx].values[0]]
          # corresponding colours
          lowest = min(shapley_values)
          highest = max(shapley_values)
          normalized, colours = [], []
          print(f'Base Value: {shap_values[0, :, selected_pred_idx].base_values}')
          print(f'Lowest: {lowest}')
          print(f'Highest: {highest}')
          for v in shapley_values:
               if v >= 0:
                    colours.append('red')
                    normalized.append(round(v / highest, 2))
               else:
                    colours.append('blue')
                    normalized.append(round(v / lowest, 2))
          # create html code for highlight
          return ''.join([self.highlighter(word, colour=colour, opacity=opacity) for word, opacity, colour in zip(self.text, normalized, colours)])

     def visualize(self, html_str: str):
          # method for jupyter lab notebooks
          display(HTML(html_str))

class ProtBertSequenceClassificationShapAnalysis(SequenceClassificationShapAnalysis):

     def __init__(self,
                  base_fp: str,
                  head_fp: str,
                  label2id: Dict[str, int], 
                  gpu_id: int):
          from PredaconDatasets.Tokenization.protbert import get_tokenizer
          tokenizer = get_tokenizer()
          super().__init__(
               base_fp=base_fp, head_fp=head_fp, tokenizer=tokenizer,
               label2id=label2id, gpu_id=gpu_id
          )

# # Example Code

# from predacons.shap.ShapAnalysis import ProtBertSequenceClassificationShapAnalysis
# from PredaconDatasets.ClassDicts.ibis import get_ec_dict

# onnx_dir = '/home/gunam/storage/deep_learning_experiments/Ibis2023/MoleculeTraining/onnx'
# base_fp = f'{onnx_dir}/ibis3_base.onnx'
# ec1_head_fp = f'{onnx_dir}/ec1_head.onnx'
# label2id = get_ec_dict(level=1)

# pipe = ProtBertSequenceClassificationShapAnalysis(
#     base_fp=base_fp,
#     head_fp=ec1_head_fp,
#     label2id=label2id,
#     gpu_id=6
# )

# text = 'MAASTTFYYPIRKSFLLPPSRNKRNPNLISCSTKPVCSPPSPSPSSLQTTSHRSQKQNLRLPSFEDSFLLYQFSSPTEDPGFSNRISEQFEGEPPELLFPSVEENKSLEISSNMWWADLKAAVGQRINVEGIVSSVSVVVRDRHLVLPHISVRDLRYIDWGELKRKGFKGVVFDKDNTLTAPYSLAIWPPLRPSIDRCKVVFGHDIAVFSNSAGLTEYDHDDSKAKALEAETGIRVLRHRVKKPAGTAEEVEKHFGCASSELIMVGDRPFTDIVYGNRNGFLTVLTEPLSRAEEPFIVRQVRRLELALLKRWLRKGLKPVDHGLVSDVTQFVKDPSDL'
# text = list(text)

# shap_values = pipe.get_shap_values(text=text)
# html = pipe.get_html(shap_values)
# pipe.visualize(html)