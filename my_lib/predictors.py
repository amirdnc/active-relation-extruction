import json

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
import  numpy as np


@Predictor.register('base-tagger')
class TaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        # for x in self._dataset_reader.proccess_json(json_dict):
        return self._dataset_reader.text_to_instance(json_dict)


@Predictor.register('siamese-tagger')
class SiamesePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        # for x in self._dataset_reader.proccess_json(json_dict):
        return self._dataset_reader.text_to_instance(json_dict)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        outputs['res'] = int(np.argmax(outputs['scores']))
        return json.dumps(outputs) + "\n"

