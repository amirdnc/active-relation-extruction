import json

from spacy.symbols import IDS
import spacy
import networkx as nx
from spacy.symbols import IDS
from tqdm import tqdm


def relation_ner(d):
    res_to_ner = {}
    ner_to_rel = {}
    for i in d:
        if i['relation'] == 'no_relation':
            continue
        s = '{} {}'.format(i['subj_type'], i['obj_type'])
        if i['relation'] not in res_to_ner:
            res_to_ner[i['relation']] = []
        res_to_ner[i['relation']].append(s)
        if s not in ner_to_rel:
            ner_to_rel[s] = []
        ner_to_rel[s].append(i['relation'])

    for r, l in res_to_ner.items():
        res_to_ner[r] = list(dict.fromkeys(l))
    for r, l in res_to_ner.items():
        print('{}: {}'.format(r, l))

    for r, l in ner_to_rel.items():
        ner_to_rel[r] = list(dict.fromkeys(l))
    for r, l in ner_to_rel.items():
        print('{}: {}'.format(r, l))

def count_relations(rel_l, d):
    counter = 0
    total = 0
    for i in d:
        if i['relation'] == 'no_relation':
            continue
        total += 1
        if i['relation'] in rel_l:
            counter += 1
    print('there are {} relations with similar ner out of {} samples, which are {}'.format(counter, total, counter / total))


def path_to_edges(path, edges):
    res = []
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) in edges:
            if(edges[(path[i], path[i + 1])]['type'] == 'compound'):
                continue
            res.append(edges[(path[i], path[i + 1])]['type'])
        else:
            if (edges[(path[i+1], path[i])]['type'] == 'compound'):
                continue
            res.append(edges[(path[i + 1], path[i])]['type'])
    return res

ROOT_ID = 1339
def fix_ent(ent):
    if ent[-1] == '.':  # if ent is ending with . it's probebly initials
        return ent
    e = nlp(ent)
    return e[0].string

def fix_sen(s):
    res = []

    for x in s.split(' '):
        if x == '-LRB-':
            res.append('(')
            continue
        if x == '-RRB-':
            res.append(')')
            continue
        res.append(x)
    return ' '.join(res)
def get_dependency(s, subj, obj):
    s = fix_sen(s)
    doc = nlp(s)
    edges = {}
    idx_word = {ROOT_ID: 'ROOT'}
    subj = fix_ent(subj.lower())
    obj = fix_ent(obj.lower())
    for token in doc:
        if token.dep_ == 'ROOT':
            edges[(token.idx, ROOT_ID)] = {'type': token.dep_}
        idx_word[token.idx] = token.string.strip().lower()
        for child in token.children:
            edges[(token.idx, child.idx)] = {'type': child.dep_}
    word_idx = {v: k for k, v in idx_word.items()}
    graph = nx.Graph(list(edges.keys()))
    nx.set_edge_attributes(graph, edges)
    try:
        path = nx.shortest_path(graph, source=word_idx[subj], target=word_idx[obj])
    except Exception as e:
        print()
        print('exception: {}'.format(e))
        print('bad sentence: {}'.format(s))
        print('subj: {}, obj:{}'.format(subj, obj))
        return None, None
    return path_to_edges(path, edges), [idx_word[x] for x in path]
    # return [reverse_ids[x] for x in path_edges]


nlp = spacy.load("en_core_web_md")


def dependency_ner(d):
    ner_dep = {}
    t = tqdm(d)
    for i in t:
        if i['relation'] == 'no_relation':
            continue
        sdp_edges, sdp_nodes = get_dependency(' '.join(i['token']), i['token'][i['subj_start']], i['token'][i['obj_start']])
        i['sdp'] = sdp_edges
        i['sdp_nodes'] = sdp_nodes
        if sdp_edges is None or len(sdp_edges) == 0:
            continue
        if i['relation'] not in ner_dep:
           ner_dep[i['relation']] = []
        ner_dep[i['relation']].append(' '.join(sdp_edges))
    for r, l in ner_dep.items():
        size = len(l)
        ner_dep[r] = list(dict.fromkeys(l))
        print('{}: {}. which is {} - {}'.format(r, size - len(ner_dep[r]), size, len(ner_dep[r])))
    for r, l in ner_dep.items():
        print('{}: {}'.format(r, l))
    return d


if '__main__' == __name__:
    path = r'/home/nlp/amirdnc/data/tacred/data/json/test.json'
    # path = r"D:\tacred\data\json\train.json"
    with open(path, 'r') as f:
        d = json.load(f)
    # relation_ner(d)
    d = dependency_ner(d)
    d = sorted(d, key=lambda x: x['relation'])
    write_path = 'dep_train.txt'
    with open(write_path, 'w') as f:
        for i in d:
            if 'sdp_nodes' not in i or i['sdp_nodes'] is None:
                continue
            f.write('\n')
            f.write('relation: {}\n'.format(i['relation']))
            f.write('text: {}\n'.format(' '.join(i['token'])))
            f.write('nodes: {}\n'.format(' '.join(i['sdp_nodes'])))
            f.write('edges: {}\n'.format(' '.join(i['sdp'])))

    with open(r'/home/nlp/amirdnc/data/tacred/data/json/test_dep.json', 'w') as f:
        d = json.dump(d, f)

    rel_l = ['per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:countries_of_residence', 'per:city_of_death', 'per:country_of_birth', 'per:employee_of', 'per:country_of_death', 'per:city_of_birth', 'per:stateorprovince_of_death', 'per:origin', 'org:stateorprovince_of_headquarters', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:member_of', 'org:subsidiaries', 'org:parents', 'per:countries_of_residence', 'per:origin', 'per:country_of_death', 'per:country_of_birth', 'per:date_of_death', 'per:date_of_birth' ,'org:stateorprovince_of_headquarters', 'org:member_of', 'org:parents', 'org:country_of_headquarters', 'org:members', 'org:member_of', 'org:parents', 'org:subsidiaries', 'per:countries_of_residence', 'per:origin', 'per:country_of_birth', 'per:country_of_death', 'per:stateorprovinces_of_residence', 'per:stateorprovince_of_birth', 'per:stateorprovince_of_death', 'per:children', 'per:siblings', 'per:spouse', 'per:other_family', 'per:alternate_names', 'per:parents', 'per:cities_of_residence', 'per:city_of_death', 'per:city_of_birth', 'org:alternate_names', 'org:member_of', 'org:members', 'org:parents', 'org:subsidiaries', 'org:shareholders', 'per:employee_of', 'per:schools_attended', 'org:founded_by', 'org:top_members/employees', 'org:shareholders']
    rel_l = list(dict.fromkeys(rel_l))
    print(len(rel_l))


    # count_relations(rel_l, d)
