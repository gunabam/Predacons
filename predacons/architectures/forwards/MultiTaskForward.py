from predacons.data.DataClass import (
    MultiInputData,
    recover_data
)
from predacons.models import DataStructs
import torch
from torch import nn
from torch.nn import ParameterDict
from typing import Union

class MultiTaskForward(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self,
                predict: bool = False,
                loss_can_be_zero: bool = False,
                **kwargs) -> DataStructs.MultiTaskOutput:
        # variables to track
        final_loss = 0
        output_dict = {}
        logits_dict = {}
        labels_dict = {}
        # run encoder and base model on batch
        if kwargs.get('batch') != None and isinstance(kwargs['batch'], MultiInputData):
            data_dict = kwargs['batch'].sentences
        else:
            data_dict = recover_data(self.inputs, **kwargs)
        for inp in self.inputs:
            output_dict[inp] = self.get_model_outputs(data_dict[inp])
        # multi task processing
        for head_name, head in self.heads.items():
            # single input classification
            if head.training_task in ['sentence_classification',
                                      'token_classification']:
                # classification tasks
                for inp in head.analyze_inputs:
                    batch = data_dict[inp]
                    if batch == None: continue
                    # name to access predictions
                    cache_name = f'{head_name}___{inp}'
                    # ground truth
                    labels = getattr(batch, head_name)
                    # input features
                    if head.training_task == 'token_classification':
                        feats = output_dict[inp].tokens
                    elif head.training_task == 'sentence_classification':
                        feats = output_dict[inp].pooled_output
                    # prediction
                    if predict == False:
                        out = head.classification(feats, labels)
                        if torch.isnan(out['loss']) == False:
                            final_loss+=out['loss']
                        if out['logits'].shape[0] > 0:
                            logits_dict[cache_name] = out['logits']
                            labels_dict[cache_name] = out['labels']
                    else:
                        logits_dict[cache_name] = head(feats)
            # pairwise (dual inputs) classification
            if head.training_task in ['siamese_sentence_classification',
                                      'siamese_token_classification']:
                for inp_1, inp_2 in head.analyze_inputs:
                    # name to access predictions
                    cache_name = f'{head_name}___{inp_1}___{inp_2}'
                    # ground truth
                    labels = kwargs.get(f'common___{cache_name}')
                    # sentence classification
                    if head.training_task == 'siamese_sentence_classification':
                        # input features
                        feats_1 = output_dict[inp_1].pooled_output
                        feats_2 = output_dict[inp_2].pooled_output
                        # prediction
                        if predict == False:
                            out = head.classification(feats_1, feats_2, labels)
                            if torch.isnan(out['loss']) == False:
                                final_loss+=out['loss']
                            if out['logits'].shape[0] > 0:
                                logits_dict[cache_name] = out['logits']
                                labels_dict[cache_name] = out['labels']
                        else:
                            logits_dict[cache_name] = head(feats_1, feats_2)
                    # token classification
                    elif head.training_task == 'siamese_token_classification':
                        # input features
                        feats_1 = output_dict[inp_1].tokens
                        feats_2 = output_dict[inp_2].tokens
                        feats_indexes_1 = getattr(output_dict[inp_1], f'{head_name}___{inp_1}')
                        feats_indexes_2 = getattr(output_dict[inp_2], f'{head_name}___{inp_2}')
                        if predict == False:
                            out = head.classification(feats_1, feats_indexes_1, feats_2, feats_indexes_2, labels)
                            if torch.isnan(out['loss']) == False:
                                final_loss+=out['loss']
                            if out['logits'].shape[0] > 0:
                                logits_dict[cache_name] = out['logits']
                                labels_dict[cache_name] = out['labels']
                        else:
                            logits_dict[cache_name] = head(feats_1, feats_indexes_1, feats_2, feats_indexes_2)          
            # contrastive loss
            if head.training_task in ['siamese_sentence_convergence',
                                      'siamese_token_convergence']:
                for inp_1, inp_2, inp_3 in head.analyze_inputs:
                    # name to access predictions
                    cache_name = f'{head_name}___{inp_1}___{inp_2}___{inp_3}'
                    # ground truth
                    labels = kwargs.get(f'common___{cache_name}')
                    if head.training_task == 'siamese_sentence_convergence':
                        feats_1 = output_dict[inp_1].pooled_output
                        feats_2 = output_dict[inp_2].pooled_output
                        feats_3 = output_dict[inp_3].pooled_output
                        # prediction
                        if predict == False:
                            out = head.convergence(feats_1, feats_2, feats_3, labels)
                            if torch.isnan(out['loss']) == False:
                                final_loss+=out['loss']
                            if out['logits'].shape[0] > 0:
                                logits_dict[cache_name] = out['logits']
                                labels_dict[cache_name] = out['labels']
                    elif head.training_task == 'siamese_token_convergence':
                        feats_1 = output_dict[inp_1].tokens
                        feats_2 = output_dict[inp_2].tokens
                        feats_3 = output_dict[inp_3].tokens
                        feats_indexes_1 = getattr(output_dict[inp_1], f'{head_name}___{inp_1}')
                        feats_indexes_2 = getattr(output_dict[inp_2], f'{head_name}___{inp_2}')
                        feats_indexes_3 = getattr(output_dict[inp_3], f'{head_name}___{inp_3}')
                        # prediction
                        if predict == False:
                            out = head.convergence(feats_1, feats_indexes_1,
                                                   feats_2, feats_indexes_2,
                                                   feats_3, feats_indexes_3,
                                                   labels)
                            if torch.isnan(out['loss']) == False:
                                final_loss+=out['loss']
                            if out['logits'].shape[0] > 0:
                                logits_dict[cache_name] = out['logits']
                                labels_dict[cache_name] = out['labels']
            # regression tasks
            if head.training_task == 'sentence_regression':
                for inp in head.analyze_inputs:
                    batch = data_dict[inp]
                    if batch == None: continue
                    # name to access predictions
                    cache_name = f'{head_name}___{inp}'
                    # ground truth
                    target = getattr(batch, head_name)
                    is_target = getattr(batch, f'{head_name}_target', None)
                    # features
                    feats = output_dict[inp].pooled_output
                    if predict == False:
                        out = head.regression(feats, target, is_target)
                        if torch.isnan(out['loss']) == False:
                            final_loss+=out['loss']
                        if out['logits'].shape[0] > 0:
                            logits_dict[cache_name] = out['logits']
                            labels_dict[cache_name] = out['labels']
                    else:
                        logits_dict[cache_name] = head(feats)
        # average loss
        if len(self.heads) > 0 and loss_can_be_zero == False:
            final_loss = final_loss / len(self.heads)
        # prepare output
        output = DataStructs.MultiTaskOutput(loss=final_loss, logits=logits_dict, labels=labels_dict)
        del data_dict
        del output_dict
        return output.split(tensor_names=self.tensor_names)