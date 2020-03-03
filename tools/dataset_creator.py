import heapq
import json
import random

import numpy as np
from scipy.spatial.ckdtree import cKDTree

ids = 0
def procces_sentance(sen_s, emb_s):
    global ids
    sen = json.loads(sen_s.split(':', 1)[1])
    emb = json.loads(emb_s.split(':', 1)[1])
    sen['embeddings'] = emb['embeddings']
    sen['ids'] = ids
    if 'indexes' in emb:
        sen['indexes'] = emb['indexes']
    ids += 1
    return sen


def close_neg(inside, outside, start):
    t = cKDTree([x['embeddings'] for x in outside])
    res = []
    for i, emb in enumerate([x['embeddings'] for x in inside]):
        neg = t.query(emb, k=4)[1]
        anc = i + start
        for i, n in enumerate(neg):
            if n >= start:  # fix index for neg
                neg[i] += len(inside)
        res.extend([(anc, x, 0) for x in neg])
    return res


def far_pos(inside, offset):
    res = []
    h = []
    for i1, p1 in enumerate([x['embeddings'] for x in inside]):
        # max_dist = 0
        # best_index = None
        for i2, p2 in enumerate([x['embeddings'] for x in inside]):
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if len(h) < 4 or h[0][0] < dist:
                if len(h) <= 4:
                    heapq.heappush(h, (dist, (i1 + offset, i2 + offset, 1)))
                else:
                    heapq.heappushpop(h, (dist, (i1 + offset, i2 + offset, 1)))
            # if dist > max_dist:
            #     max_dist = dist
            #     best_index = (i1 + offset, i2 + offset, 1)
        res.extend([x[1] for x in h])
    return res

def far_pos(inside, offset, size):
    res = []
    for i1 in range(len(inside)):
        new_points = random.sample(range(0, len(inside)), min(size, len(inside)))
        res.extend([(i1+ offset, x+offset, 1) for x in new_points])
    return res

def close_neg(inside, outside, start, size):
    res = []
    for i in range(len(inside)):
        neg = random.sample(range(0, len(outside)), size)
        anc = i + start
        for i, n in enumerate(neg):
            if n >= start:  # fix index for neg
                neg[i] += len(inside)
        res.extend([(anc, x, 0) for x in neg])

    return res


def get_sentance(sen):
    sentance = {}
    sentance['relation'] = sen['relation']
    if 'indexes' in sentance:
        sentance['indexes'] = sen['indexes']
    sentance['token'] = sen['token']
    sentance['subj_start'] = sen['subj_start']
    sentance['subj_end'] = sen['subj_end']
    sentance['obj_start'] = sen['obj_start']
    sentance['obj_end'] = sen['obj_end']
    return sentance

def createdataset(indxs, res):
    return [(get_sentance(res[i[0]]), get_sentance(res[i[1]]), i[2]) for i in indxs]


def createdataset2(indxs, res):
    out =[(res[i[0]], res[i[1]], int(res[i[0]]['relation'] == res[i[1]]['relation']))
          for i in indxs]
    c = 0
    for s1, s2, l in out:
        # if not (s1['relation'] == s2['relation']) == bool(l):
        #     c += 1
        #     print('error {}'.format(c))
        assert (s1['relation'] == s2['relation']) == bool(l)
    return out

def get_dataset(dataset):
    'return a data set sorted by relation types'
    with open(dataset, 'r') as f:
        l1, l2 = f.readline(), f.readline()
        f.readline()  # remove blank line
        res = []
        while l1:
            res.append(procces_sentance(l1, l2))  # first line is sentence, second is the emmbeddings
            l1, l2 = f.readline(), f.readline()
            f.readline()  #remove blank line
        return sorted(res, key=lambda x: x['relation'])

def get_json_dataset(dataset):
    with open(dataset, 'r') as f:
        d = json.load(f)
    return sorted(d, key=lambda x: x['relation'])


def get_realtion_offset(relations):
    cur_l = relations[0]['relation']
    rels_idx= {}
    start_idx = 0
    for i, sen in enumerate(relations):
        if cur_l != sen['relation']:
            rels_idx[cur_l] = (start_idx, i -1)
            start_idx = i
            cur_l = sen['relation']
    rels_idx[cur_l] = (start_idx, len(relations) - 1)
    return rels_idx
def create_pair_dataset(res, rels_idx, test_relations = None):
    indxs = []
    for rel, (start, end) in rels_idx.items():
        print('processing relation {}'.format(rel))
        outside = res[:start] + res[end:]
        inside = res[start: end]
        if rel == 'no_relation':
            indxs.extend(close_neg(inside, outside, start, 3))
            continue
        indxs.extend(close_neg(inside, outside, start, num_neg))
        if not test_relations or rel in test_relations:  # in test, take only the relevant as positive
            indxs.extend(far_pos(inside, start, num_pos))
    indxs = [x for x in indxs if x is not None]
    print('len: {}'.format(len(indxs)))
    sentances = res  #[get_sentance(x) for x in res]
    data = createdataset2(indxs, sentances)
    random.shuffle(data)
    return data

def remove_relation(data, rels_idx, rel, max_allowed = 0):
    rel_len = rels_idx[rel][1] - rels_idx[rel][0] + 1 - max_allowed#len(rel_data)
    rel_start = rels_idx[rel][0]
    for rel_cur, (start, end) in rels_idx.items():
        if start > rel_start:
            rels_idx[rel_cur] = (start - rel_len, end - rel_len)
    new_data = data[:rels_idx[rel][0]] +data[rels_idx[rel][0]:rels_idx[rel][0] + max_allowed] +  data[rels_idx[rel][1]+1:]
    if max_allowed == 0:
        del rels_idx[rel]
    else:
        rels_idx[rel] = (rel_start, rel_start + max_allowed - 1)
    return new_data, rels_idx
num_pos = 3
num_neg = 3

all_rels = ['org:alternate_names', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:dissolved', 'org:founded', 'org:founded_by', 'org:member_of', 'org:members', 'org:number_of_employees/members', 'org:parents', 'org:political/religious_affiliation', 'org:shareholders', 'org:stateorprovince_of_headquarters', 'org:subsidiaries', 'org:top_members/employees', 'org:website', 'per:age', 'per:alternate_names', 'per:cause_of_death', 'per:charges', 'per:children', 'per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death', 'per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:date_of_birth', 'per:date_of_death', 'per:employee_of', 'per:origin', 'per:other_family', 'per:parents', 'per:religion', 'per:schools_attended', 'per:siblings', 'per:spouse', 'per:stateorprovince_of_birth', 'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence', 'per:title']
# remove no_relation

def unite_dataset():
    path_tacred = r'/home/nlp/amirdnc/data/tacred/data/json/train.json'
    path_docred = r'/home/nlp/amirdnc/data/docred/tacred_format/train_distant.json'
    path_out = r'/home/nlp/amirdnc/data/docred_tacred/train_united.json'
    d = []
    with open(path_tacred, 'r') as f:
        d.extend(json.load(f))
    with open(path_docred, 'r') as f:
        d.extend(json.load(f))
    random.shuffle(d)
    with open(path_out, 'w') as f:
        json.dump(d, f)
    print("done")


if __name__ == '__main__':
    unite_dataset()
    exit()
    task = 'dev'
    # dataset_path = ['../{}_out.json'.format(task)]
    dataset_path = ['/home/nlp/amirdnc/data/tacred/data/json/{}.json'.format(task)]
    relations_for_train = random.sample(all_rels, 3)
    relations_for_train = ['per:country_of_death', 'per:charges', 'per:schools_attended']
    print(relations_for_train)
    out_path = '{}{}_{}_few.json'.format(task, num_pos, '_'.join(relations_for_train).replace(':','-'))
    do_train = 'train' == task
    res = []
    for path in dataset_path:
        res.extend(get_json_dataset(path))
    rels_idx = get_realtion_offset(res)
    res_size = len(res)
    if do_train:
        for r in relations_for_train:
            res, rels_idx = remove_relation(res, rels_idx, r, 3)
    print('removed {} for the original {}'.format(res_size - len(res), res_size))
    if do_train:
        data = create_pair_dataset(res, rels_idx)
    else:
        data = create_pair_dataset(res, rels_idx, relations_for_train)
    len([x for x in data if x[2]==1]) / len(data)
    with open(out_path, 'w') as outfile:
        json.dump(data, outfile)
    print('saved to {}.'.format(out_path))
