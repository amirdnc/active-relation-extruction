import json
import shutil
import sys

from allennlp.commands import main

# config_file = "siamese_config.jsonnet"
config_file = "supervised_config.jsonnet"

# Use overrides to train on CPU.
# overrides = json.dumps({"trainer":{"cuda_device": -1},"iterator": {"type": "basic", "batch_size": 2,"instances_per_epoch":4}})
#on_the_fly
# overrides = json.dumps({"validation_data_path": "data/dev_NOTA_100.json","trainer":{"cuda_device": -1},"iterator": {"type": "basic", "batch_size": 2,"instances_per_epoch":2}})
#
overrides = json.dumps({"train_data_path": "/home/nlp/amirdnc/data/tacred/data/json/train.json","trainer":{"cuda_device": 0},
  "validation_data_path": "/home/nlp/amirdnc/data/tacred/data/json/dev.json","iterator": {"type": "basic", "batch_size": 2,"instances_per_epoch":2},
                        "validation_iterator": {"type": "basic",
                            "batch_size": 2,
                            "instances_per_epoch": 2
                        }, "dataset_reader": {"debug": "True"}, "validation_dataset_reader": {"debug": "True"}})

overrides = json.dumps({"dataset_reader": {"debug": "True"}, "validation_dataset_reader": {"debug": "True"}})
#
#four way
# overrides = json.dumps({"trainer":{"cuda_device": -1},
#   "validation_data_path": "data/dev_NOTA_100.json","iterator": {"type": "basic", "batch_size": 4,"instances_per_epoch":4},
#                         "validation_iterator": {"type": "basic",
#                             "batch_size": 4,
#                             "instances_per_epoch": 20
#                         },
#                         "dataset_reader":{"size":100}
#                         })

# overrides = json.dumps({"train_data_path": "data/train_small.json",
#   "validation_data_path": "data/val_small.json"})
CUDA_LAUNCH_BLOCKING=1
serialization_dir = "/tmp/debug_mtb/"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_lib",
    "-o", overrides,
]
# sys.argv = [
#     "allennlp",  # command name, not used by main
#     "predict",
#     "results/dropout_after_first_large_hidden/",
#     r'/home/nlp/amirdnc/code/active-relation-extruction/data/tacred-test.json',
#     "--predictor", "base-tagger",
#     "--include-package", "my_lib",
#     "--cuda-device", "0",
#     "-o", overrides,
# ]

# sys.argv = [
#      'allennlp',
#      'predict', 'results/siamese_4_shuffle', 'data/tacred-siamese-dev1.txt', '--include-package',
#      'my_lib', '--predictor', 'siamese-tagger', '--cuda-device', '0'
# ]

# sys.argv = 'allennlp predict results/supervised_two_basic single.json  --include-package my_lib --predictor base-tagger --cuda-device 0'.split()
# sys.argv = [
#      'allennlp',
#      'evaluate', 'results/siamese_rand', '/home/nlp/amirdnc/data/siamese/dev4_rand.json', '--include-package',
#      'my_lib', '--cuda-device', '0'
# ]

main()