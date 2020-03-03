import json

from tqdm import tqdm

get_relation_pos = lambda t, loc: doc['vertexSet'][l[t]]['pos'][loc]
def proccess_doc(doc):
    res = []
    for l in doc['labels']:
        if len(l['evidence']) == 0:
            l['evidence'] = list(dict.fromkeys([x['sent_id'] for x in doc['vertexSet'][l['h']]] + [x['sent_id'] for x in doc['vertexSet'][l['t']]]))
        if len(l['evidence']) != 1:
            continue
        subj_data = doc['vertexSet'][l['h']]
        obj_data = doc['vertexSet'][l['t']]
        # if doc['title'] == 'ABBA Live':
        #     print(doc['title'])
        try:
            obj_data = [x for x in obj_data if x['sent_id'] == l['evidence'][0]][0]
            subj_data = [x for x in subj_data if x['sent_id'] == l['evidence'][0]][0]
        except:
            print('Error parsing in {}'.format(doc['title']))
            continue
        temp = {}
        temp['token'] = doc['sents'][l['evidence'][0]]
        temp['subj_start'] = subj_data['pos'][0]
        temp['subj_end'] = subj_data['pos'][1]
        temp['subj_type'] = subj_data['type']
        temp['obj_type'] = obj_data['type']
        temp['obj_start'] = obj_data['pos'][0]
        temp['obj_end'] = obj_data['pos'][1]
        temp['relation'] = l['r']
        res.append(temp)
    return res


if __name__ == '__main__':
    in_dir = r'/home/nlp/amirdnc/data/docred/'
    out_dir = r'/home/nlp/amirdnc/data/docred/tacred_format/'
    in_file = r'train_annotated.json'
    in_file = r'train_distant.json'
    in_file = r'dev.json'
    # path = r'/home/nlp/amirdnc/data/docred/dev.json'
    in_path = in_dir + in_file
    out_path = out_dir + in_file
    with open(in_path, 'r') as f:
        d = json.load(f)
    output = []
    t = tqdm(d)
    for doc in t:
        output.extend(proccess_doc(doc))
    print('writing {} relations'.format(len(output)))
    with open(out_path, 'w') as f:
        d = json.dump(output, f)
