import json


def create_new_file(source, target):
    with open(source, 'r') as f:
        data = json.load(f)
    with open(target, 'w') as f:
        f.writelines([json.dumps(l) + '\n' for l in data])
    print('done')

if __name__ == '__main__':
    source = r'/home/nlp/amirdnc/data/tacred/data/json/train.json'
    # source = '../../../data/siamese/dev1.json'
    target = '../data/tacred-train-for-prediction.txt'
    create_new_file(source, target)