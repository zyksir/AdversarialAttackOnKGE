# proposed by "Data Poisoning Attack against Knowledge Graph Embedding"
# we use the Direct Attack in the paper
# we want to find the triple (h', r', t') = argmax(f(h,r',t') - f(h+dh, r', t'))
# CUDA_VISIBLE_DEVICES=0 python codes/noise_generator/direct_addition.py --init_checkpoint ./models/ComplEx_FB15k-237_baseline/

import itertools

import torch

from collections import defaultdict
from random_noise import *
import torch.autograd as autograd


class DirectAddition(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(DirectAddition, self).__init__(args)
        self.score_func = lambda s1, s2: args.lambda1 * s1 - args.lambda2 * s2
        self.name = "direct"

        self.true_rel_head, self.true_rel_tail = defaultdict(set), defaultdict(set)
        for triple in self.input_data.all_true_triples:
            self.add_true_triple(triple)
    
    def add_true_triple(self, triple):
        h, r, t = triple
        self.true_rel_tail[h].add((r, t))
        self.true_rel_head[t].add((r, h))

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        args = self.args
        h, r, t = test_triple
        true_cand = self.true_rel_tail[h] if mode == "head-batch" else self.true_rel_head[t]
        s = time.time()
        cand_r_list = random.choices(self.all_relations, k=args.num_cand)
        cand_e_list = random.choices(self.all_entities, k=args.num_cand)
        cand_r_e_list = list(set(zip(cand_r_list, cand_e_list)).difference(true_cand))
        cand_r_list, cand_e_list = zip(*cand_r_e_list)
        cand_r_list, cand_e_list = list(cand_r_list), list(cand_e_list)
        args.num_cand = len(cand_r_list)

        embed_h = self.kge_model.entity_embedding[h]
        embed_r = self.kge_model.relation_embedding[r]
        embed_t = self.kge_model.entity_embedding[t]
        score = self.kge_model.score_embedding(embed_h, embed_r, embed_t)
        perturbed_embed_h, perturbed_embed_t = None, None
        if mode == "head-batch":
            embed_h_grad = autograd.grad(score, embed_h)[0]
            perturbed_embed_h = embed_h - args.epsilon * embed_h_grad
        elif mode == "tail-batch":
            embed_t_grad = autograd.grad(score, embed_t)[0]
            perturbed_embed_t = embed_t - args.epsilon * embed_t_grad

        b_begin = 0
        cand_scores = []
        with torch.no_grad():
            while b_begin < args.num_cand:
                b_cand_r = cand_r_list[b_begin: b_begin + args.num_cand]
                b_cand_e = cand_e_list[b_begin: b_begin + args.num_cand]
                b_begin += args.num_cand

                embed_cand_r = self.kge_model.relation_embedding[b_cand_r]
                embed_cand_e = self.kge_model.entity_embedding[b_cand_e]
                s1, s2 = None, None
                if mode == "head-batch":
                    s1 = self.kge_model.score_embedding(perturbed_embed_h, embed_cand_r, embed_cand_e, mode=mode)
                    s2 = self.kge_model.score_embedding(embed_h, embed_cand_r, embed_cand_e, mode=mode)
                elif mode == "tail-batch":
                    s1 = self.kge_model.score_embedding(embed_cand_e, embed_cand_r, perturbed_embed_t, mode=mode)
                    s2 = self.kge_model.score_embedding(embed_cand_e, embed_cand_r, embed_t, mode=mode)
                score = self.score_func(s1, s2)
                score = score.detach().cpu().numpy().tolist()
                cand_scores += score
        cand_scores = np.array(cand_scores)
        idx = np.argmax(cand_scores)
        score = cand_scores[idx]
        if mode == "head-batch":
            return (h, cand_r_list[idx], cand_e_list[idx]), score.item()
        return (cand_e_list[idx], cand_r_list[idx], t), score.item()

    def get_noise_triples(self):
        noise_triples, args = self.noise_triples, self.args
        args.num_cand = np.math.ceil((args.nentity*args.nrelation)*args.corruption_factor / 100)
        all_true_triples = set(self.input_data.all_true_triples)
        for i in range(len(self.target_triples)):
            sys.stdout.write("%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            target_triple = self.target_triples[i]
            noise_triple_h, score_h = self.get_noise_for_head(target_triple, mode="head-batch")
            noise_triple_t, score_t = self.get_noise_for_head(target_triple, mode="tail-batch")
            if score_h > score_t:
                noise_triples.add(noise_triple_h)
                self.add_true_triple(noise_triple_h)
            else:
                noise_triples.add(noise_triple_t)
                self.add_true_triple(noise_triple_t)
        return list(noise_triples)

class CentralDiffAddition(DirectAddition):
    def __init__(self, args):
        super(CentralDiffAddition, self).__init__(args)
        self.name = "central_diff"
        self.args.epsilon = self.args.learning_rate

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        args = self.args
        h, r, t = test_triple
        true_cand = self.true_rel_tail[h] if mode == "head-batch" else self.true_rel_head[t]
        cand_r_list = random.choices(self.all_relations, k=args.num_cand)
        cand_e_list = random.choices(self.all_entities, k=args.num_cand)
        cand_r_e_list = list(set(zip(cand_r_list, cand_e_list)).difference(true_cand))
        cand_r_list, cand_e_list = zip(*cand_r_e_list)
        cand_r_list, cand_e_list = list(cand_r_list), list(cand_e_list)
        args.num_cand = len(cand_r_list)

        embed_h = self.kge_model.entity_embedding[h]
        embed_r = self.kge_model.relation_embedding[r]
        embed_t = self.kge_model.entity_embedding[t]
        score = self.kge_model.score_embedding(embed_h, embed_r, embed_t)
        perturbed_embed_e, enforced_embed_e = None, None
        ########## begin difference ############
        if mode == "head-batch":
            embed_h_grad = autograd.grad(score, embed_h)[0]
            perturbed_embed_e = embed_h - args.epsilon * embed_h_grad
            enforced_embed_e = embed_h + args.epsilon * embed_h_grad
        elif mode == "tail-batch":
            embed_t_grad = autograd.grad(score, embed_t)[0]
            perturbed_embed_e = embed_t - args.epsilon * embed_t_grad
            enforced_embed_e = embed_t + args.epsilon * embed_t_grad
        ########## end difference ############

        b_begin = 0
        cand_scores = []
        while b_begin < args.num_cand:
            b_cand_r = cand_r_list[b_begin: b_begin + args.num_cand]
            b_cand_e = cand_e_list[b_begin: b_begin + args.num_cand]
            b_begin += args.num_cand

            embed_cand_r = self.kge_model.relation_embedding[b_cand_r]
            embed_cand_e = self.kge_model.entity_embedding[b_cand_e]
            s1, s2 = None, None
            ########## begin difference ############
            if mode == "head-batch":
                s1 = self.kge_model.score_embedding(perturbed_embed_e, embed_cand_r, embed_cand_e, mode=mode)
                s2 = self.kge_model.score_embedding(enforced_embed_e, embed_cand_r, embed_cand_e, mode=mode)
            elif mode == "tail-batch":
                s1 = self.kge_model.score_embedding(embed_cand_e, embed_cand_r, perturbed_embed_e, mode=mode)
                s2 = self.kge_model.score_embedding(embed_cand_e, embed_cand_r, enforced_embed_e, mode=mode)
            ########## end difference ############
            score = self.score_func(s1, s2)
            score = score.detach().cpu().numpy().tolist()
            cand_scores += score
        cand_scores = np.array(cand_scores)
        idx = np.argmax(cand_scores)
        score = cand_scores[idx]
        if mode == "head-batch":
            return (h, cand_r_list[idx], cand_e_list[idx]), score.item()
        return (cand_e_list[idx], cand_r_list[idx], t), score.item()

class DirectRelAddition(DirectAddition):
    def __init__(self, args):
        super(DirectRelAddition, self).__init__(args)
        self.score_func = lambda s1, s2: args.lambda1 * s1 - args.lambda2 * s2
        self.name = "direct_rel"
        self.true_head_tail = {}
        for h, r, t in self.input_data.all_true_triples:
            if r not in self.true_head_tail:
                self.true_head_tail[r] = set()
            self.true_head_tail[r].add((h, t))

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        if mode == "tail-batch":
            return test_triple, -1e9
        args = self.args
        h, r, t = test_triple
        s = time.time()
        true_cand = self.true_head_tail[r]
        cand_h_list = random.choices(self.all_entities, k=args.num_cand)
        cand_t_list = random.choices(self.all_entities, k=args.num_cand)
        cand_h_t_list = list(set(zip(cand_h_list, cand_t_list)).difference(true_cand))
        cand_h_list, cand_t_list = zip(*cand_h_t_list)
        cand_h_list, cand_t_list = list(cand_h_list), list(cand_t_list)
        args.num_cand = len(cand_h_list)
        e1 = time.time()

        embed_h = self.kge_model.entity_embedding[h]
        embed_r = self.kge_model.relation_embedding[r]
        embed_t = self.kge_model.entity_embedding[t]
        score = self.kge_model.score_embedding(embed_h, embed_r, embed_t)
        embed_r_grad = autograd.grad(score, embed_r)[0]
        perturbed_embed_r = embed_r - args.epsilon * embed_r_grad
        e2 = time.time()

        b_begin = 0
        cand_scores = []
        with torch.no_grad():
            while b_begin < args.num_cand:
                b_cand_h = cand_h_list[b_begin: b_begin + args.num_cand]
                b_cand_t = cand_t_list[b_begin: b_begin + args.num_cand]
                b_begin += args.num_cand

                embed_cand_h = self.kge_model.entity_embedding[b_cand_h]
                embed_cand_t = self.kge_model.entity_embedding[b_cand_t]
                s1 = self.kge_model.score_embedding(embed_cand_h, perturbed_embed_r, embed_cand_t, mode=mode)
                s2 = self.kge_model.score_embedding(embed_cand_h, embed_r, embed_cand_t, mode=mode)
                score = self.score_func(s1, s2)
                score = score.detach().cpu().numpy().tolist()
                cand_scores += score
        cand_scores = np.array(cand_scores)
        idx = np.argmax(cand_scores)
        score = cand_scores[idx]
        e3 = time.time()
        self.true_head_tail[r].add((cand_h_list[idx], cand_t_list[idx]))
        return (cand_h_list[idx], r, cand_t_list[idx]), score.item()

if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    
    suffix = ""
    if args.corruption_factor != 5:
        suffix = "_%d" % args.corruption_factor
    generator = DirectAddition(args)
    generator.generate("direct" + suffix)
    
    generator = CentralDiffAddition(args)
    generator.generate("central_diff" + suffix)
    
    generator = DirectRelAddition(args)
    generator.generate("direct_rel")