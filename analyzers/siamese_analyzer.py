import json
import random

import numpy as np
# plt.figure(figsize=[20, 20], dpi=800)
from analyzers.utils import read_predict_file, plt_heatmap

#
# def plt_heatmap(mat, label_dict: dict):
#     labels = [x[1] for x in sorted(label_dict.items())]
#     data = mat
#
#     fig, ax = plt.subplots()
#     ax.pcolor(data)
#     ax.axis('tight')
#     ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
#            yticks=np.arange(len(labels)), yticklabels=labels)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5, forward=True)
#     plt.show()
rels = ['no_relation', 'org:alternate_names', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:dissolved', 'org:founded', 'org:founded_by', 'org:member_of', 'org:members', 'org:number_of_employees/members', 'org:parents', 'org:political/religious_affiliation', 'org:shareholders', 'org:stateorprovince_of_headquarters', 'org:subsidiaries', 'org:top_members/employees', 'org:website', 'per:age', 'per:alternate_names', 'per:cause_of_death', 'per:charges', 'per:children', 'per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death', 'per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:date_of_birth', 'per:date_of_death', 'per:employee_of', 'per:origin', 'per:other_family', 'per:parents', 'per:religion', 'per:schools_attended', 'per:siblings', 'per:spouse', 'per:stateorprovince_of_birth', 'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence', 'per:title']


def procces_sentance(sen_s, emb_s):
    sen = json.loads(sen_s.split(':', 1)[1])
    results = json.loads(emb_s.split(':', 1)[1])
    line = {}
    line['r1'] = sen[0]['relation']
    line['r2'] = sen[1]['relation']
    line['pred'] = np.argmax(results['scores'])
    line['label'] = sen[2]
    line['correct'] = line['label'] == line['pred']
    return line


def create_pair_dataset(dataset):
    lines = read_predict_file(dataset, procces_sentance)
    random.shuffle(lines)
    rel_to_int = {r: i for i,r in enumerate(rels)}
    int_to_rel = {i: r for i,r in enumerate(rels)}
    good = np.zeros([len(int_to_rel),len(int_to_rel)])
    overall = np.zeros([len(int_to_rel),len(int_to_rel)])
    for l in lines:
        r1 = rel_to_int[l['r1']]
        r2 = rel_to_int[l['r2']]
        overall[r1, r2] += 1
        overall[r2, r1] += 1
        if l['correct']:
            good[r2, r1] += 1
            good[r1, r2] += 1
    acc = good/overall
    print(rels)
    print(acc)
    plt_heatmap(overall, int_to_rel)



if __name__ == '__main__':
    dataset_path = '../siamese_ner_out.txt'
    create_pair_dataset(dataset_path)