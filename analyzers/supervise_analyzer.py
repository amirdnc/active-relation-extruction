import json
import numpy as np
import stanfordnlp

from analyzers.utils import read_predict_file, plt_heatmap


def proccess_sentence(input, pred):
    sen = json.loads(input.split(':', 1)[1])
    results = json.loads(pred.split(':', 1)[1])
    line = {}
    line['token'] = sen['token']
    line['subj'] = sen['subj_type']
    line['obj'] = sen['obj_type']
    line['r'] = sen['relation']
    line['scores'] = results['scores']
    line['label'] = results['label']
    line['obj_start'] = sen['obj_start']
    line['subj_start'] = sen['subj_start']
    int_to_rel[line['label']] = sen['relation']
    rel_to_int[line['r']] = results['label']
    line['pred'] = np.argmax(line['scores'])
    return line

def relation_heatmap(d, int_to_rel):
    counter = 0
    tot = 0
    for l in d:
        if l['r'] == 'no_relation':
            continue
        tot += 1
        if l['pred'] == l['label']:
            counter += 1
    print(float(counter) / tot)

    mat = np.zeros([len(int_to_rel), len(int_to_rel)])
    for l in d:
        mat[l['pred']][l['label']] += 1
    for i in range(len(int_to_rel)):
        s = np.sum(mat[i])
        for j in range(len(int_to_rel)):
            mat[i][j] = mat[j][i] / s
    plt_heatmap(mat, int_to_rel, 'normalized supervised')


def format_sentence(tokens):
    s = ' '.join([x for x in tokens if not x.startswith('[')])
    s = s.replace('-LRB-', '(')
    return s.replace('-RRB-', ')')

if __name__ == '__main__':
    path = r'../ner_real_out1.txt'
    int_to_rel = {}
    rel_to_int = {}
    d = read_predict_file(path, proccess_sentence)

    nlp = stanfordnlp.Pipeline()
    for l in d:
        if l['pred'] != l['label']:
            print('predicted: ', int_to_rel[l['pred']])
            print('real: ', l['r'])
            print(' '.join(l['token']))
            raw= format_sentence(l['token'])

            print(raw)
            n = nlp(raw)
            t_tokens = [dep_edge[2].text for dep_edge in n.sentences[0].dependencies]
            temp = [(dep_edge[2].text, dep_edge[0].index, t_tokens[int(dep_edge[0].index)-1], i)for dep_edge, i in zip(n.sentences[0].dependencies, range(1, len(n.sentences[0].dependencies)+1))]
            raw_l = raw.split(' ')
            s_word = raw_l[l['subj_start']]
            o_word = raw_l[l['obj_start']]
            s_index = raw_l.index(s_word)
            o_index = raw_l.index(o_word)
            try:
                s_dep = temp[s_index][2]
                o_dep = temp[o_index][2]
                print('subj dependency:{}, obj dependency:{}'.format(s_dep, o_dep))
            except:
                print('error computing dependency')

            # print('subj dependency: {}'.format(raw[loc]))