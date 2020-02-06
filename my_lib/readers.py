from typing import Dict
from typing import List
import json
import logging
import os
import random
# from preprocessing_prepeare_sentence import preprocessing
# from preprocessing_prepeare_sentence import head_start_token,head_end_token,tail_start_token,tail_end_token
import copy
from overrides import overrides
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ListField, IndexField, MetadataField, Field



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

e1_start = '[unused1]'
e1_end = '[unused2]'
e2_start = '[unused3]'
e2_end = '[unused4]'
default_ner = '[unused5]'

UNUSED = '[unused{}]'
class NerDict():
    def __init__(self, ner_dict_path):
        self.path = ner_dict_path
        if os.path.isfile(ner_dict_path):
            with open(ner_dict_path) as f:
                data = json.load(f)
                self.d = data['d']
                self.counter = data['counter']
        else:
            self.d = {}
            self.counter = 6  # first unused

    def get_ner_token(self, ne):
        if isinstance(ne, list):
            if len(ne) > 1:
                print("size of list is{}".len(ne))
            ne = ne[0]
        if ne not in self.d:
            self.d[ne] = UNUSED.format(self.counter)
            self.counter += 1
            with open(self.path, 'w') as f:
                json.dump({'d': self.d, 'counter': self.counter}, f)

        return self.d[ne]


def add_word_markers(line):
    # line['token'] = [start_line] + line['token'] + [end_line]
    if line['subj_start'] < line['obj_start']:
        indexes = [line['subj_start'], line['subj_end'] + 2, line['obj_start'] + 2, line['obj_end'] + 4]
        line['token'].insert(indexes[0], e1_start)
        line['token'].insert(indexes[1], e1_end)
        line['token'].insert(indexes[2], e2_start)
        line['token'].insert(indexes[3], e2_end)
    else:
        indexes = [line['subj_start'] + 2, line['subj_end'] + 4, line['obj_start'], line['obj_end'] + 2]
        line['token'].insert(indexes[2], e2_start)
        line['token'].insert(indexes[3], e2_end)
        line['token'].insert(indexes[0], e1_start)
        line['token'].insert(indexes[1], e1_end)
    line['indexes'] = indexes
    return line


def add_mask_markers(line):
    indexes = [line['subj_start'], line['subj_end'] + 1, line['obj_start'], line['obj_end'] + 1]
    if line['subj_start'] < line['obj_start']:
        line['token'] = line['token'][:line['obj_start']] + [e2_start, default_ner, e2_end] + line['token'][
                                                                                              line['obj_end'] + 1:]
        line['token'] = line['token'][:line['subj_start']] + [e1_start, default_ner, e1_end] + line['token'][
                                                                                               line['subj_end'] + 1:]
    else:
        line['token'] = line['token'][:line['subj_start']] + [e1_start, default_ner, e1_end] + line['token'][line['subj_end'] + 1:]
        line['token'] = line['token'][:line['obj_start']] + [e2_start, default_ner, e2_end] + line['token'][line['obj_end'] + 1:]
    return line

def add_ner_markers(line, nd):
    # indexes = [line['subj_start'], line['subj_end'] + 1, line['obj_start'], line['obj_end'] + 1]
    if line['subj_start'] < line['obj_start']:
        line['token'] = line['token'][:line['obj_start']] + [e2_start, nd.get_ner_token(line['obj_type']), e2_end] + line['token'][
                                                                                              line['obj_end'] + 1:]
        line['token'] = line['token'][:line['subj_start']] + [e1_start, nd.get_ner_token(line['subj_type']), e1_end] + line['token'][
                                                                                               line['subj_end'] + 1:]
    else:
        line['token'] = line['token'][:line['subj_start']] + [e1_start, nd.get_ner_token(line['subj_type']), e1_end] + line['token'][line['subj_end'] + 1:]
        line['token'] = line['token'][:line['obj_start']] + [e2_start, nd.get_ner_token([line['obj_type']]), e2_end] + line['token'][line['obj_end'] + 1:]
    return line


# def get_ner_mask(ner):


def load_annotated_tokens(infile, debug=False):
    data = json.load(infile)
    if debug:
        data = data[:500]  # for debugging
    # adjusted_tokens = [add_new_word_markers(x) for x in data if x['relation'] == 'no_relation']
    # adjusted_tokens = [add_new_word_markers(x) for x in data]
    return data




@DatasetReader.register("tacred_reader")
class TacredReader(DatasetReader):
    def __init__(self,
                 bert_model: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None, debug=False, mode='basic') -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.spacy_splitter = SpacyWordSplitter(keep_spacy_tokens=True)
        self.TRAIN_DATA = "meta_train"
        self.TEST_DATA = "meta_test"
        self.tokens_with_markers = "tokens_with_markers"
        self.head_bert = "head_after_bert"
        self.tail_bert = "tail_after_bert"
        self.labels_to_id = {}
        self.max_id = 0
        self.debug = debug
        self.marker_tokens = self._tokenizer.tokenize('{} {}'.format(e1_start, e2_start))
        self.mode = mode
        if mode =='ner':
            self.ner_dict = NerDict('ner_dict.json')
    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from json files at: %s", data_file)
            data = json.load(data_file)
            if self.debug:
                data = data[:500]  # for debugging
            for x in data:
                yield self.text_to_instance(x)

    def load_ner_dict(self, ner_dict_path):
        if os.path.isfile(ner_dict_path):
            with open(ner_dict_path) as f:
                return json.load(f)
        else:
            return {}
    @overrides
    def text_to_instance(self, data: dict) -> Instance:  # type: ignore
        if self.mode == 'basic':
            data = add_word_markers(data)
        elif self.mode == 'ner':
            data = add_ner_markers(data, self.ner_dict)
        elif self.mode == 'mask':
            data = add_mask_markers(data)

        tokenized_tokens = self._tokenizer.tokenize(' '.join(data['token']))
        field_of_tokens = TextField(tokenized_tokens, self._token_indexers)
        marker_loc = [tokenized_tokens.index(x)+1 for x in self.marker_tokens] # +1 for the ['cls'] token
        return Instance({'sen':field_of_tokens, 'loc': MetadataField(marker_loc), 'label': LabelField(data['relation'])})  # , 'sentence': TextField(' '.join(data['token']))})

    def proccess_json(self, json_data):
        return add_word_markers(json_data)

    def get_id(self, label):
        if label not in self.labels_to_id:
            self.labels_to_id[label] = self.max_id
            self.max_id += 1
        return self.labels_to_id[label]


@DatasetReader.register("siamese_reader")
class SiameseReader(TacredReader):
    def __init__(self,
                 bert_model: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None, debug=False) -> None:
        super().__init__(bert_model, lazy, tokenizer, token_indexers, debug)

    def tokenize(self, d):
        tokenized_tokens = self._tokenizer.tokenize(' '.join(d['token']))
        return TextField(tokenized_tokens, self._token_indexers)

    @overrides
    def text_to_instance(self, data: dict) -> Instance:  # type: ignore
        s0 = data[0]
        s1 = data[1]
        l0 = [s0['indexes'][0], s0['indexes'][2]]
        l1 = [s1['indexes'][0], s1['indexes'][2]]
        return Instance({'sen1':self.tokenize(data[0]), 'sen2': self.tokenize(data[1]), 'loc': ListField([MetadataField(l0), MetadataField(l1)]), 'label': LabelField(data[2], skip_indexing=True)})  # , 'sentence': TextField(' '.join(data['token']))})}

@DatasetReader.register("multi_ner_reader")
class MultiNERReader(TacredReader):
    def __init__(self,
                 bert_model: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None, debug=False) -> None:
        super().__init__(bert_model, lazy, tokenizer, token_indexers, debug)
        self.ner_dict = NerDict('ner_dict.json')

    @overrides
    def text_to_instance(self, data: dict) -> Instance:  # type: ignore
        sens = []
        base_tokens = data['token']
        sens.append(add_word_markers(data)['token'])
        data['token'] = base_tokens
        sens.append(add_ner_markers(data, self.ner_dict)['token'])
        data['token'] = base_tokens
        sens.append(add_mask_markers(data)['token'])

        markers = []
        texts = []
        for s in sens:
            tokenized_tokens = self._tokenizer.tokenize(' '.join(s))
            texts.append(TextField(tokenized_tokens, self._token_indexers))
            markers.append(MetadataField([tokenized_tokens.index(x) + 1 for x in self.marker_tokens]))  # +1 for the ['cls'] token

        return Instance({'sen0': texts[0], 'sen1': texts[1], 'sen2': texts[2], 'loc0': markers[0], 'loc1': markers[1], 'loc2': markers[2],
                         'label': LabelField(data['relation'])})  # , 'sentence': TextField(' '.join(data['token']))})
