import json


if '__main__' == __name__:
    path = r'/home/nlp/amirdnc/data/docred/train_annotated.json'
    path = r'/home/nlp/amirdnc/data/docred/train_distant.json'
    path = r'/home/nlp/amirdnc/data/docred/dev.json'
    with open(path, 'r') as f:
        d = json.load(f)
    tot = 0
    one_s = 0
    for i in d:
        tot += len(i['labels'])
        for l in i['labels']:
            if len(l['evidence']) == 0:
                l['evidence'] = list(dict.fromkeys([x['sent_id'] for x in i['vertexSet'][l['h']]] + [x['sent_id'] for x in i['vertexSet'][l['h']]]))
            if len(l['evidence']) == 1:
                one_s +=1
    print('there are {} single sentences relation out {} relations, which are {}'.format(one_s, tot, one_s / tot))
