from typing import Dict, List
import collections
import logging
import math
import allennlp
import torch
from overrides import overrides
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from torch.autograd import Variable
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from torch.nn.parameter import Parameter
from allennlp.nn import util
# from my_library.models.my_loss_metric import SpecialLoss, NotaNotInsideBest2
from my_lib.metrics import SpecialLoss

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


BERT_LARGE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                     "hidden_act": "gelu",
                     "hidden_dropout_prob": 0.1,
                     "hidden_size": 1024,
                     "initializer_range": 0.02,
                     "intermediate_size": 4096,
                     "max_position_embeddings": 512,
                     "num_attention_heads": 16,
                     "num_hidden_layers": 24,
                     "type_vocab_size": 2,
                     "vocab_size": 30522
                    }

BERT_BASE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "type_vocab_size": 2,
                    "vocab_size": 30522
                   }
linear = "linear"

NER_NUM_LABELS = 17
@Model.register('relation_clasification')
class RelationClassification(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers : int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.2,
                 devices = 0,
                 num_labels = 42):

        super().__init__(vocab,regularizer)
        self.num_labels = num_labels
        self.embbedings = text_field_embedder
        self.bert_type_model = BERT_BASE_CONFIG
        self.extractor = EndpointSpanExtractor(input_dim=self.bert_type_model['hidden_size'], combination="x,y")
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        if isinstance(devices, list):
            devices = devices[0]
        if devices:
            self.device = torch.device("cuda:{}".format(devices) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda")
        self.metrics = {
            # "NOTA_NotInBest2": NotaNotInsideBest2(),
            "accuracy": CategoricalAccuracy(),
            'f1': SpecialLoss(0)  # F1Measure(1)  # no relation is 0
        }
        # for i in range(1,42):
        #     self.metrics['f1_{}'.format(i)] = F1Measure(i)
        self.first_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'] * 2,hidden_dim)
        self.second_liner_layer = torch.nn.Linear(hidden_dim, self.num_labels)  # TACRED labels

        self.number_of_linear_layers = number_of_linear_layers
        self.tanh = torch.nn.Tanh()
        self.drop_layer = torch.nn.Dropout(p=drop_out_rate)
        self.counter = 0

    def one_hot(self, v, size):
        res = torch.zeros(torch.tensor(v.size()), size).to(self.device)
        return res.scatter_(1, v.view(-1, 1), 1)

    @overrides
    def forward(self, sen, loc, clean_tokens = None, test_clean_text = None,
                label = None) -> Dict[str, torch.Tensor]:

        bert_context_for_relation = self.embbedings(sen)
        bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, loc)

        scores, embeddings = self.go_thorugh_mlp(bert_represntation, self.first_liner_layer, self.second_liner_layer)
        scores = scores.to(self.device)
        output_dict = {}
        # output_dict['embeddings'] = embeddings
        # output_dict['indexes'] = loc
        if label is not None:
            loss = self.crossEntropyLoss(scores, label)
            output_dict["loss"] = loss
            self.counter += 1
        self.metrics['accuracy'](scores, label)
        for k in self.metrics.keys():
            self.metrics[k](scores, label)
        output_dict["scores"] = scores.view(-1, self.num_labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        res = self.metrics['f1'].get_metric(reset)
        res = {"accuracy": res[0], 'recall': res[1], "f1": res[2]}
        # for i in range(1,42):
        #     res['f1_{}'.format(i)] = self.metrics['f1_{}'.format(i)].get_metric(reset)
        return res

    def go_thorugh_mlp(self, concat_represntentions, first_layer, second_layer):
        # return self.relation_layer_norm(concat_represntentions)
        concat_represntentions = concat_represntentions.to(self.device)
        after_drop_out_layer = concat_represntentions
        after_first_layer = self.drop_layer(first_layer(after_drop_out_layer))
        x = self.tanh(after_first_layer)
        x = second_layer(x)
        return x, after_drop_out_layer

    def extract_vectors_from_markers(self, embbeds, location):
        stacked_embed = embbeds.view(-1,embbeds.size(-2), embbeds.size(-1)).to(self.device)
        pt_tensor_from_list = torch.FloatTensor(location)
        indeces = util.combine_initial_dims(pt_tensor_from_list).long().to(self.device)
        value = self.extractor(stacked_embed, indeces)
        return value

@Model.register('relation_clasification_one_layer')
class RelationClassificationSingleLayer(RelationClassification):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers : int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.2,
                 devices=0,
                 num_labels=42
                 ):
        super().__init__(vocab, text_field_embedder, number_of_linear_layers, skip_connection, regularizer, hidden_dim, add_distance_from_mean, drop_out_rate, devices, num_labels)
        self.first_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'] * 2, self.num_labels)

    @overrides
    def go_thorugh_mlp(self, concat_represntentions,first_layer, second_layer):
        # return self.relation_layer_norm(concat_represntentions)
        concat_represntentions = concat_represntentions.to(self.device)
        after_drop_out_layer = self.drop_layer(concat_represntentions)
        after_first_layer = first_layer(after_drop_out_layer)
        return after_first_layer, concat_represntentions


@Model.register('siamese_sentences')
class SiameseRelations(RelationClassification):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers : int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.4,
                 devices=0
                 ):
        super().__init__(vocab, text_field_embedder, number_of_linear_layers, skip_connection, regularizer, hidden_dim, add_distance_from_mean, drop_out_rate, devices)
        self.first_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'] * 2, self.bert_type_model['hidden_size'] * 2)
        self.second_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'] * 4, 2)

    @overrides
    def go_thorugh_mlp(self, concat_represntentions,first_layer,second_layer):
        # return self.relation_layer_norm(concat_represntentions)
        concat_represntentions = concat_represntentions.to(self.device)
        after_drop_out_layer = self.drop_layer(concat_represntentions)
        after_first_layer = first_layer(after_drop_out_layer)
        return after_first_layer


    def get_embeddings(self, sen, loc):
        bert_context_for_relation = self.embbedings(sen)
        bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, loc)
        concat_represntentions = bert_represntation.to(self.device)
        after_drop_out_layer = self.drop_layer(concat_represntentions)
        after_first_layer = self.tanh(self.first_liner_layer(after_drop_out_layer))
        return after_first_layer

    def s1_embedding(self, sen1, loc):
        return {'embeddings': self.get_embeddings()}
    @overrides
    def forward(self, sen1, sen2, loc, clean_tokens=None, test_clean_text=None,
                label=None) -> Dict[str, torch.Tensor]:
        if not sen2:
            return self.s1_embedding(sen1, loc)
        loc = list(zip(*loc))
        emb0 = self.get_embeddings(sen1, loc[0])
        emb1 = self.get_embeddings(sen2, loc[1])
        x = torch.cat((emb0, emb1), 1)
        scores = self.second_liner_layer(x)
        scores = scores.to(self.device)
        output_dict = {}
        # logger.info('label = {}'.format(label))
        if label is not None:
            loss = self.crossEntropyLoss(scores, label)
            output_dict["loss"] = loss
            self.counter += 1
        self.metrics['accuracy'](scores, label)
        self.metrics['f1'](scores, label)
        for k in self.metrics.keys():
            self.metrics[k](scores, label)
        output_dict["scores"] = scores
        return output_dict


@Model.register('siamese_ner_sentences')
class SiameseNerRelations(SiameseRelations):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers : int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.4,
                 devices=0
                 ):
        super().__init__(vocab, text_field_embedder, number_of_linear_layers, skip_connection, regularizer, hidden_dim, add_distance_from_mean, drop_out_rate, devices)
        self.first_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'] * 2,
                                                 self.bert_type_model['hidden_size'] * 2)
        self.second_liner_layer = torch.nn.Linear((self.bert_type_model['hidden_size'] + NER_NUM_LABELS) * 4, self.bert_type_model['hidden_size'] * 2)
        self.third_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'] * 2, 2)

    @overrides
    def go_thorugh_mlp(self, concat_represntentions, first_layer, second_layer):
        # return self.relation_layer_norm(concat_represntentions)
        concat_represntentions = concat_represntentions.to(self.device)
        after_drop_out_layer = self.drop_layer(concat_represntentions)
        after_first_layer = self.tanh(first_layer(after_drop_out_layer))
        scores = second_layer(after_first_layer)
        return scores
    @overrides
    def forward(self, sen1, sen2, loc, clean_tokens=None, test_clean_text=None,
                label=None, ner=None) -> Dict[str, torch.Tensor]:
        loc = list(zip(*loc))
        emb0 = self.get_embeddings(sen1, loc[0])
        emb1 = self.get_embeddings(sen2, loc[1])

        ner_embd = [self.one_hot(x.to(self.device), NER_NUM_LABELS) for x in torch.tensor(list(zip(*ner)))]
        x = torch.cat([emb0, emb1] + ner_embd, 1)
        scores = self.go_thorugh_mlp(x, self.second_liner_layer, self.third_liner_layer)
        scores = scores.to(self.device)
        output_dict = {}
        # logger.info('label = {}'.format(label))
        if label is not None:
            loss = self.crossEntropyLoss(scores, label)
            output_dict["loss"] = loss
            self.counter += 1
        self.metrics['accuracy'](scores, label)
        self.metrics['f1'](scores, label)
        for k in self.metrics.keys():
            self.metrics[k](scores, label)
        output_dict["scores"] = scores
        return output_dict

@Model.register('relation_clasification_multi_types')
class RelationClassificationSMultiTypes(RelationClassification):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers : int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.2,
                 devices = 0):
        super().__init__(vocab, text_field_embedder, number_of_linear_layers, skip_connection, regularizer, hidden_dim, add_distance_from_mean, drop_out_rate, devices)
        self.hidden_size = self.bert_type_model['hidden_size'] * 2* 3
        self.first_liner_layer = torch.nn.Linear(self.hidden_size, self.bert_type_model['hidden_size'])
        self.second_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size'], self.num_labels)
        self.drop_layer2d = torch.nn.Dropout2d(0.2)
        self.batch_norm_layer = torch.nn.BatchNorm1d(self.hidden_size)
    def forward(self, sen0, sen1, sen2, loc0, loc1, loc2, clean_tokens = None, test_clean_text = None,
                label = None) -> Dict[str, torch.Tensor]:

        bert_context_for_relation0 = self.embbedings(sen0)
        bert_context_for_relation1 = self.embbedings(sen1)
        bert_context_for_relation2 = self.embbedings(sen2)
        num_samples = bert_context_for_relation0.size(0)
        ex = self.extract_vectors_from_markers
        representaions = [ex(bert_context_for_relation0, loc0), ex(bert_context_for_relation1, loc1), ex(bert_context_for_relation2, loc2)]
        concate_rep = torch.stack(representaions, dim = 2)
        # norm_rep = self.batch_norm_layer(concate_rep.view(num_samples, self.hidden_size))
        # concate_rep = norm_rep.view(num_samples*3, int(self.hidden_size / 3))
        final_rep = self.drop_layer2d(concate_rep)
        dim_final = final_rep.view(num_samples, self.hidden_size)
        scores, embeddings = self.go_thorugh_mlp(dim_final, self.first_liner_layer, self.second_liner_layer)
        scores = scores.to(self.device)
        output_dict = {}
        output_dict['embeddings'] = embeddings
        if label is not None:
            loss = self.crossEntropyLoss(scores, label)
            output_dict["loss"] = loss
            self.counter += 1
        self.metrics['accuracy'](scores, label)
        for k in self.metrics.keys():
            self.metrics[k](scores, label)
        output_dict["scores"] = scores.view(-1, self.num_labels)
        return output_dict

@Model.register('relation_ner_clasifications')
class RelationNerClassification(RelationClassification):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers: int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.2,
                 devices=0):
        super().__init__(vocab, text_field_embedder, number_of_linear_layers, skip_connection, regularizer,
                         hidden_dim, add_distance_from_mean, drop_out_rate, devices)
        self.ner_layer1 = torch.nn.Linear(self.bert_type_model['hidden_size'], hidden_dim)
        self.ner_layer2 = torch.nn.Linear(hidden_dim, NER_NUM_LABELS)

    def classify_data(self, represntation, l1, l2, label, output_dict, metric):
        scores, _ = self.go_thorugh_mlp(represntation, l1, l2)
        scores = scores.to(self.device)
        if label is not None:
            loss = self.crossEntropyLoss(scores, label)
            output_dict["loss"] += loss
            metric(scores, label)
        return output_dict, scores
    @overrides
    def forward(self, sen, loc, clean_tokens=None, test_clean_text=None,
                label=None, ner=None) -> Dict[str, torch.Tensor]:

        bert_context_for_relation = self.embbedings(sen)
        bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, loc)
        s_rep, o_rep = bert_represntation.split(self.bert_type_model['hidden_size'], dim=1)
        s_labels = torch.tensor(list(zip(*ner))[0]).to(self.device)
        o_labels = torch.tensor(list(zip(*ner))[1]).to(self.device)
        output_dict = {'loss': 0}
        output_dict, scores = self.classify_data(bert_represntation, self.first_liner_layer, self.second_liner_layer, label, output_dict, self.metrics['f1'])
        output_dict, scores = self.classify_data(s_rep, self.ner_layer1,
                                                 self.ner_layer2, s_labels, output_dict, self.metrics['accuracy'])
        output_dict, scores = self.classify_data(s_rep, self.ner_layer1,
                                                 self.ner_layer2, o_labels, output_dict, self.metrics['accuracy'])
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = super().get_metrics(reset)
        d['ner_accuracy'] = self.metrics['accuracy'].get_metric(reset)
        return d

@Model.register('relation_ner_clasifications_with_embedding')
class RelationNerClassificationWithEmmbedding(RelationNerClassification):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers: int = 2,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.2,
                 devices=0):
        super().__init__(vocab, text_field_embedder, number_of_linear_layers, skip_connection, regularizer,
                         hidden_dim, add_distance_from_mean, drop_out_rate, devices)

        self.first_liner_layer = torch.nn.Linear((self.bert_type_model['hidden_size'] + NER_NUM_LABELS)* 2,hidden_dim)
        self.ner_layer1 = torch.nn.Linear(self.bert_type_model['hidden_size'], hidden_dim)
        self.ner_layer2 = torch.nn.Linear(hidden_dim, NER_NUM_LABELS)

    @overrides
    def forward(self, sen, loc, clean_tokens=None, test_clean_text=None,
                label=None, ner=None) -> Dict[str, torch.Tensor]:
        bert_context_for_relation = self.embbedings(sen)
        bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, loc)
        # s_rep, o_rep = bert_represntation.split(self.bert_type_model['hidden_size'], dim=1)
        s_labels = torch.tensor(list(zip(*ner))[0]).to(self.device)
        o_labels = torch.tensor(list(zip(*ner))[1]).to(self.device)
        output_dict = {'loss': 0}
        # output_dict, s_scores = self.classify_data(s_rep, self.ner_layer1,
        #                                          self.ner_layer2, s_labels, output_dict,
        #                                          self.metrics['accuracy'])
        # output_dict, o_scores = self.classify_data(s_rep, self.ner_layer1,
        #                                          self.ner_layer2, o_labels, output_dict,
        #                                          self.metrics['accuracy'])
        s_scores = self.one_hot(s_labels, NER_NUM_LABELS)
        o_scores = self.one_hot(o_labels, NER_NUM_LABELS)
        relation_rep = torch.cat([bert_represntation, s_scores, o_scores], dim=1)
        output_dict, scores = self.classify_data(relation_rep, self.first_liner_layer,
                                                 self.second_liner_layer, label, output_dict,
                                                 self.metrics['f1'])
        output_dict['scores'] = scores
        output_dict['label'] = label

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = super().get_metrics(reset)
        d['ner_accuracy'] = self.metrics['accuracy'].get_metric(reset)
        return d
