import json


def create_new_file(source, target):
    with open(source, 'r') as f:
        data = json.load(f)
    with open(target, 'w') as f:
        f.writelines([json.dumps(l) + '\n' for l in data])
    print('done')

if __name__ == '__main__':
    source = '/home/nlp/sharos/span_bert/SpanBERT/permut_ALL_wiki_pure_exlusive_pred/data/json/test_24-33.json'
    # source = '../../../data/siamese/dev1.json'
    target = r'/home/nlp/amirdnc/data/shahar_temp.json'
    # target = '../data/siamese_testte4_rand_ner.txt'
    create_new_file(source, target)