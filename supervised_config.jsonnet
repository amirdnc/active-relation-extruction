local cuda = 2;
//local cuda = [3];
local bert_type = 'bert-base-cased';
local mode = 'basic';
local train_data = {
"tacred_base": "/home/nlp/amirdnc/data/tacred/data/json/train.json",
"docred_base": "/home/nlp/amirdnc/data/docred/tacred_format/train_distant.json",
"doc_tac": "/home/nlp/amirdnc/data/docred_tacred/train_united.json",
};

local dev_data = {
"tacred_base": "/home/nlp/amirdnc/data/tacred/data/json/dev.json",
"docred_base": "/home/nlp/amirdnc/data/docred/tacred_format/dev.json",
"doc_tac": "/home/nlp/amirdnc/data/tacred/data/json/dev.json",
};
local num_labels = {
"tacred_base": 42,
"docred_base": 96,
"doc_tac": 138
};
//local bert_type = 'bert-large-cased';
local batch_size = 9;
local setup = "doc_tac";
local lr_with_find = 0.00002;
local instances_per_epoch = 100;
//local reader = "relation_ner_reader";
local reader = "tacred_reader";
{
  "dataset_reader": {
    "type": reader,
    "bert_model": bert_type,
    "lazy": true,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "my-bert-basic-tokenizer",
        "do_lower_case": false
      }
    },
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_type,
          "do_lowercase": false,
          "use_starting_offsets": false
      }
    },
    "mode": mode,
  },
    "validation_dataset_reader": {
    "type": reader,
    "bert_model": bert_type,
    "lazy": false,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "my-bert-basic-tokenizer",
        "do_lower_case": false
      }
    },
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_type,
          "do_lowercase": false,
          "use_starting_offsets": false
      }
    },
    "mode": mode,
  },
  "train_data_path": train_data[setup] ,
  "validation_data_path": dev_data[setup],
  "model": {
    //"type": "relation_ner_clasifications_with_embedding",
    //"type": "relation_ner_clasifications",
    "type": "relation_clasification_one_layer",
    //"type": "relation_clasification_multi_types",
    "hidden_dim": 768,
    "add_distance_from_mean": true,
    "number_of_linear_layers": 2,
    "drop_out_rate": 0.2,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert"]
        },
        "token_embedders": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model":  bert_type,
              "top_layer_only": true,
              "requires_grad": true
            }
        },
    },
    "regularizer": [["liner_layer", {"type": "l2", "alpha": 1e-02}], [".*", {"type": "l2", "alpha": 1e-07}]],
    "devices": cuda,
    "num_labels": num_labels[setup],
  },
  "iterator": {
    "type": "shuffle",
    "batch_size": batch_size,
    "instances_per_epoch": null
  },
    "validation_iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": null
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
//        "type": "sgd",
//        "momentum": 0.05,
        "lr": lr_with_find,
        "weight_decay": 0.0,
    },
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+f1",
    "num_epochs": 15,
    "patience": 10,
    "cuda_device": cuda
  }
}