from _ast import Dict

import allennlp
import overrides
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import Parameter

class RelationClassification(Model):
    def __init__(self, temp: float = 0.2):
        pass
