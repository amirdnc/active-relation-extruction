from collections import defaultdict

import spacy
import networkx as nx
from spacy.symbols import IDS

nlp = spacy.load("en_core_web_sm")


def path_to_edges(path, edges):
    res = []
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) in edges:
            res.append(edges[(path[i], path[i + 1])]['type'])
        else:
            res.append(edges[(path[i + 1], path[i])]['type'])
    return res



if '__main__' == __name__:
    s = 'Lomax shares a story about Almena Lomax , his mother and a newspaper owner and journalist in Los Angeles , taking her family on the bus to Tuskegee , Ala. , in 1961 .'
    s = 'Many did default , including Prachai , the founder of Thai Petrochemical Industry , to the tune of US$ 2.7 billion .'
    s = "John's friend, Josh, lives in New-York"
    s= "She moved to Sao Paulo, where the baby, a boy, was born in February."
    s = "Merck in mid-July completed its -LRB- EURO -RRB- 5.2 billion acquisition of biotech equipment maker Millipore Corp. , based in Billerica , Massachusetts ."
    s = " In addition , the International Agency for Research on Cancer -LRB- IARC -RRB- has concluded that there is sufficient evidence that p - dichlorobenzene is carcinogenic in treated animals -LSB- IARC 1987 -RSB- ."
    s= "U.S. Undersecretary of State Nicholas Burns told reporters Thursday that Washington wants a third U.N. Security Council sanctions resolution passed as soon as possible ."
    doc = nlp(s)
    reverse_ids = {v: k for k,v in IDS.items()}
    # Load spacy's dependency tree into a networkx graph
    edges = {}  # defaultdict(list)
    idx_word = {}
    for token in doc:
        idx_word[token.idx] = token.string.strip().lower()
        for child in token.children:
            edges[(token.idx, child.idx)] = {'type': child.dep_}

    word_idx = {v: k for k, v in idx_word.items()}
    print(edges)
    graph = nx.Graph(list(edges.keys()))
    nx.set_edge_attributes(graph, edges)
    # Get the length and path
    entity1 = word_idx['nicholas'.lower()]
    entity2 = word_idx['u.s.'.lower()]
    # print(nx.shortest_path_length(graph, source=entity1, target=entity2))
    path = nx.shortest_path(graph, source=entity1, target=entity2)
    path_edges = path_to_edges(path, edges)
    print([idx_word[int(x)] for x in path])
    print(path_edges)