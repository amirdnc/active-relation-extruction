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

def far_pos(inside, offset):
    res = []
    for i1, p1 in enumerate([x['embeddings'] for x in inside]):
        new_points = random.sample(range(0, len(inside)), min(4, len(inside)))
        res.extend([(i1+ offset, x+offset, 1) for x in new_points])
    return res

def close_neg(inside, outside, start):
    res = []
    for i, emb in enumerate([x['embeddings'] for x in inside]):
        neg = random.sample(range(0, len(outside)), 4)
        anc = i + start
        for i, n in enumerate(neg):
            if n >= start:  # fix index for neg
                neg[i] += len(inside)
        res.extend([(anc, x, 0) for x in neg])
    return res


def get_sentance(sen):
    sentance = {}
    sentance['relation'] = sen['relation']
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
    # temp = []
    # for i in indxs:
    #     # print(i)
    #     temp.append((res[i[0]], res[i[1]], i[2]))
    return [(res[i[0]], res[i[1]], i[2]) for i in indxs]


def create_pair_dataset(dataset):
    with open(dataset, 'r') as f:
        l1, l2 = f.readline(), f.readline()
        f.readline()  # remove blank line
        res = []
        while l1:
            res.append(procces_sentance(l1, l2))  # first line is sentence, second is the emmbeddings
            l1, l2 = f.readline(), f.readline()
            f.readline()  #remove blank line
            # if len((res)) > 1000:
            #     break
        res = sorted(res, key=lambda x: x['relation'])
        cur_l = res[0]['relation']
        rels_idx= {}
        start_idx = 0
        for i, sen in enumerate(res):
            if cur_l != sen['relation']:
                rels_idx[cur_l] = (start_idx, i)
                start_idx = i + 1
                cur_l = sen['relation']
        rels_idx[cur_l] = (start_idx, len(res) - 1)
        indxs = []
        for rel, (start, end) in rels_idx.items():
            print('processing relation {}'.format(rel))
            outside = res[:start] + res[end:]
            inside = res[start: end]
            indxs.extend(close_neg(inside, outside, start))
            if rel == 'no_relation':
                continue
            indxs.extend(far_pos(inside, start))
        print('len before: {}'.format(len(indxs)))
        indxs = [x for x in indxs if x is not None]
        print('len after: {}'.format(len(indxs)))
        sentances = [get_sentance(x) for x in res]
        # data = createdataset(indxs, res)
        data = createdataset2(indxs, sentances)
        random.shuffle(data)
        with open('train4_rand_fin.json', 'w') as outfile:
            json.dump(data, outfile)

    print('done')


num_pos = 4
num_neg = 4
if __name__ == '__main__':
    dataset_path = '../data/train_out.json'
    create_pair_dataset(dataset_path)