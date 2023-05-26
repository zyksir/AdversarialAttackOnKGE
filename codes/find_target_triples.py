# Target: find which triples should we attack
# python find_target_triples.py wn18rr
# python find_target_triples.py FB15k-237
import os
import sys
import pickle
from utils import read_triple

# for DATASET in ["FB15k-237", "wn18rr"]:
DATASET = "FB15k-237" # sys.argv[1]
if DATASET == "FB15k-237":
    target_rank = 1
elif DATASET == "wn18rr":
    target_rank = 5
else:
    raise Exception("dataset should be either FB15k-237 or wn18rr")
data_path = f"../data/{DATASET}"
print(f"generate target triples for DATASET {DATASET}")

# load dataset info
with open(os.path.join(data_path, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join(data_path, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)
train_triples = read_triple(os.path.join(data_path, "train.txt"), entity2id, relation2id)
valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
all_true_triples = train_triples + valid_triples + test_triples

entity_density = {}
for head, relation, tail in all_true_triples:
    if head not in entity_density:
        entity_density[head] = 0
    if tail not in entity_density:
        entity_density[tail] = 0
    entity_density[head] += 1
    entity_density[tail] += 1

# load testing results
model2triple2rank = {}
model2top1Triples = {}
for MODEL in ["TransE", "RotatE", "DistMult", "ComplEx"]:
    triple2rank_path = f"../models/{MODEL}_{DATASET}_baseline/triple2ranking.pkl"
    with open(triple2rank_path, "rb") as f:
        triple2rank = pickle.load(f)
    model2triple2rank[MODEL] = triple2rank
    model2top1Triples[MODEL] = set()
    for triple, mode2ranking in triple2rank.items():
        rankh, rankt = mode2ranking["head-batch"], mode2ranking["tail-batch"]
        if rankh <= target_rank and rankt <= target_rank:
            model2top1Triples[MODEL].add(triple)

# load 
top1Triples = model2top1Triples["TransE"]\
    .intersection(model2top1Triples["RotatE"], model2top1Triples["DistMult"], model2top1Triples["ComplEx"])

top1Triples_score = [((head, relation, tail), entity_density[head], entity_density[tail]) for head, relation, tail in top1Triples]
top1Triples_score = sorted(top1Triples_score, key=lambda x: -(x[1] + x[2]))[:100]
targetTriples = [triple for triple, _, _ in top1Triples_score]
with open(os.path.join(data_path, 'targetTriples.pkl'), "wb") as fw:
    print(f"get {len(targetTriples)} targetTriples")
    pickle.dump(targetTriples, fw)
