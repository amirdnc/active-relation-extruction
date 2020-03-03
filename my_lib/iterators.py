from collections import deque
from typing import Iterable, Deque
import logging
import random

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)

@DataIterator.register("shuffle")
class ShuffleIterator(BasicIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.
    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool):
        # First break the dataset into memory-sized lists:
        list_of_instances = list(instances)
        random.shuffle(list_of_instances)
        return super()._create_batches(instances, shuffle)