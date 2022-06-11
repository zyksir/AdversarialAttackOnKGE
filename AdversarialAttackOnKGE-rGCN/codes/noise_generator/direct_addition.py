# proposed by "Data Poisoning Attack against Knowledge Graph Embedding"
# we use the Direct Attack in the paper
# we want to find the triple (h', r', t') = argmax(f(h,r',t') - f(h+dh, r', t'))
# CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/direct_addition.py --init_checkpoint ./models/RotatE_FB15k-237_baseline

import itertools

import torch

from random_noise import *
import torch.autograd as autograd


class DirectAddition(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(DirectAddition, self).__init__(args)
        self.score_func = lambda s1, s2: args.lambda1 * s1 - args.lambda2 * s2
        self.name = "direct"

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        args = self.args
        h, r, t = test_triple
        cand_r_list = random.choices(self.all_relations, k=args.num_cand)
        cand_e_list = random.choices(self.all_entities, k=args.num_cand)
        cand_r_e_list = list(set(zip(cand_r_list, cand_e_list)))
        cand_r_list, cand_e_list = zip(*cand_r_e_list)
        cand_r_list, cand_e_list = list(cand_r_list), list(cand_e_list)
        args.num_cand = len(cand_r_list)

        embed_h = self.kge_model.node_embeddings[h]
        embed_r = self.kge_model.scoring_function.relations[r]
        embed_t = self.kge_model.node_embeddings[t]
        train = get_input_data(args).train_triples
        graph = torch.tensor(train, dtype=torch.long).to('cuda:0')
        test_triple = torch.tensor(test_triple, dtype=torch.long).unsqueeze(0).to('cuda:0')

        x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
        score, _, _ = self.kge_model(graph, test_triple, x)
        perturbed_embed_h, perturbed_embed_t = None, None
        if mode == "head-batch":
            embed_h_grad = autograd.grad(score, x)[0][h]
            perturbed_embed_h = embed_h - args.epsilon * embed_h_grad
        elif mode == "tail-batch":
            embed_t_grad = autograd.grad(score, x)[0][t]
            perturbed_embed_t = embed_t - args.epsilon * embed_t_grad

        b_begin = 0
        cand_scores = []
        with torch.no_grad():
            while b_begin < args.num_cand:
                b_cand_r = cand_r_list[b_begin: b_begin + args.num_cand]
                b_cand_e = cand_e_list[b_begin: b_begin + args.num_cand]
                b_begin += args.num_cand
                
                embed_cand_r = self.kge_model.scoring_function.relations[b_cand_r]
                embed_cand_e = self.kge_model.node_embeddings[b_cand_e]
                s1, s2 = None, None
                if mode == "head-batch":
                    can_triples = []
                    for i in range(args.num_cand):
                      can_triples.append([h, b_cand_r[i], b_cand_e[i]])
                    can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                    x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
                    x_perturbed = x
                    x_perturbed[h] += perturbed_embed_h
            
                    s1, _, _ = self.kge_model(graph, can_triples, x_perturbed)
                    s2, _, _ = self.kge_model(graph, can_triples, x)
                elif mode == "tail-batch":
                    can_triples = []
                    for i in range(args.num_cand):
                      can_triples.append([b_cand_e[i], b_cand_r[i], t])
                    can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                    x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
                    x_perturbed = x
                    x_perturbed[t] += perturbed_embed_t

                    s1, _, _ = self.kge_model(graph, can_triples, x_perturbed)
                    s2, _, _ = self.kge_model(graph, can_triples, x)
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
        noise_triples, args = set(), self.args
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
            else:
                noise_triples.add(noise_triple_t)
        print("len of noise triples: %d" % len(noise_triples))
        print("len of true triples: %d"% len(noise_triples.intersection(all_true_triples)))
        return list(noise_triples)


class TaylorAddition(DirectAddition):
    def __init__(self, args):
        super(TaylorAddition, self).__init__(args)
        self.score_func = lambda s1, s2: args.lambda1 * s1 * 1.0 / args.lambda2 * s2
        self.name = "taylor"


class CentralDiffAddition(DirectAddition):
    def __init__(self, args):
        super(CentralDiffAddition, self).__init__(args)
        self.name = "central_diff"

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        args = self.args
        h, r, t = test_triple
        cand_r_list = random.choices(self.all_relations, k=args.num_cand)
        cand_e_list = random.choices(self.all_entities, k=args.num_cand)
        cand_r_e_list = list(set(zip(cand_r_list, cand_e_list)))
        cand_r_list, cand_e_list = zip(*cand_r_e_list)
        cand_r_list, cand_e_list = list(cand_r_list), list(cand_e_list)
        args.num_cand = len(cand_r_list)

        embed_h = self.kge_model.node_embeddings[h]
        embed_r = self.kge_model.scoring_function.relations[r]
        embed_t = self.kge_model.node_embeddings[t]
        train = get_input_data(args).train_triples
        graph = torch.tensor(train, dtype=torch.long).to('cuda:0')
        test_triple = torch.tensor(test_triple, dtype=torch.long).unsqueeze(0).to('cuda:0')

        x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
        score, _, _ = self.kge_model(graph, test_triple, x)
        perturbed_embed_e, enforced_embed_e = None, None
        ########## begin difference ############
        if mode == "head-batch":
            embed_h_grad = autograd.grad(score, x)[0][h]
            perturbed_embed_e = embed_h - args.epsilon * embed_h_grad
            enforced_embed_e = embed_h + args.epsilon * embed_h_grad
        elif mode == "tail-batch":
            embed_t_grad = autograd.grad(score, x)[0][t]
            perturbed_embed_e = embed_t - args.epsilon * embed_t_grad
            enforced_embed_e = embed_t + args.epsilon * embed_t_grad
        ########## end difference ############

        b_begin = 0
        cand_scores = []
        while b_begin < args.num_cand:
            b_cand_r = cand_r_list[b_begin: b_begin + args.num_cand]
            b_cand_e = cand_e_list[b_begin: b_begin + args.num_cand]
            b_begin += args.num_cand

            embed_cand_r = self.kge_model.scoring_function.relations[b_cand_r]
            embed_cand_e = self.kge_model.node_embeddings[b_cand_e]
            s1, s2 = None, None
            ########## begin difference ############
            if mode == "head-batch":
                can_triples = []
                for i in range(args.num_cand):
                  can_triples.append([h, b_cand_r[i], b_cand_e[i]])
                can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
                x_perturbed = x
                x_perturbed[h] += perturbed_embed_e
                x_enforced = x
                x_enforced[h] += enforced_embed_e

                s1, _, _ = self.kge_model(graph, can_triples, x_perturbed)
                s2, _, _ = self.kge_model(graph, can_triples, x_enforced)
            elif mode == "tail-batch":
                can_triples = []
                for i in range(args.num_cand):
                  can_triples.append([b_cand_e[i], b_cand_r[i], t])
                can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
                x_perturbed = x
                x_perturbed[t] += perturbed_embed_e
                x_enforced = x
                x_enforced[t] += enforced_embed_e

                s1, _, _ = self.kge_model(graph, can_triples, x_perturbed)
                s2, _, _ = self.kge_model(graph, can_triples, x_enforced)
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
        super(DirectAddition, self).__init__(args)
        self.score_func = lambda s1, s2: args.lambda1 * s1 - args.lambda2 * s2
        self.true_head, self.true_tail = {}, {}
        for h, r, t in self.input_data.all_true_triples:
            if (h, r) not in self.true_tail:
                self.true_tail[(h, r)] = set()
            if (r, t) not in self.true_head:
                self.true_head[(r, t)] = set()
            self.true_tail[(h, r)].add(t)
            self.true_head[(r, t)].add(h)
        self.name = "direct_rel"

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        if mode == "tail-batch":
            return test_triple, -1e9
        args = self.args
        h, r, t = test_triple
        args.num_cand_ent = np.math.ceil(args.nentity / 100)
        # embed()

        cand_e_list = random.sample(self.all_entities, k=args.num_cand_ent)
        cand_h_t_list = list(itertools.product(cand_e_list, cand_e_list))
        cand_h_list, cand_t_list = zip(*cand_h_t_list)
        cand_h_list, cand_t_list = list(cand_h_list), list(cand_t_list)
        args.num_cand = len(cand_h_list)

        embed_h = self.kge_model.node_embeddings[h]
        embed_r = self.kge_model.scoring_function.relations[r]
        embed_t = self.kge_model.node_embeddings[t]
        train = get_input_data(args).train_triples
        graph = torch.tensor(train, dtype=torch.long).to('cuda:0')
        test_triple = torch.tensor(test_triple, dtype=torch.long).unsqueeze(0).to('cuda:0')

        x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
        score, _, embed_r_grad = self.kge_model(graph, test_triple, x)
        perturbed_embed_r = embed_r - args.epsilon * embed_r_grad

        b_begin = 0
        cand_scores = []
        with torch.no_grad():
            while b_begin < args.num_cand:
                b_cand_h = cand_h_list[b_begin: b_begin + args.num_cand]
                b_cand_t = cand_t_list[b_begin: b_begin + args.num_cand]
                b_begin += args.num_cand

                embed_cand_h = self.kge_model.node_embeddings[b_cand_h]
                embed_cand_t = self.kge_model.node_embeddings[b_cand_t]
                can_triples = []
                for i in range(args.num_cand):
                  can_triples.append([b_cand_h[i], r, b_cand_t[i]])
                can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias

                s1, _, _ = self.kge_model(graph, can_triples, x, perturb_r=True, pertubation=perturbed_embed_r)
                s2, _, _ = self.kge_model(graph, can_triples, x)
                score = self.score_func(s1, s2)
                score = score.detach().cpu().numpy().tolist()
                cand_scores += score
        cand_scores = np.array(cand_scores)
        idx = np.argmax(cand_scores)
        score = cand_scores[idx]
        return (cand_h_list[idx], r, cand_t_list[idx]), score.item()

if __name__ == "__main__":
    args = get_noise_args()
    #override_config(args)
    suffix = ""
    if args.corruption_factor != 5:
        suffix = "_%d" % args.corruption_factor

    generator = DirectAddition(args)
    generator.generate("direct" + suffix)

    generator = TaylorAddition(args)
    generator.generate("taylor" + suffix)

    generator = CentralDiffAddition(args)
    generator.generate("central_diff" + suffix)

    generator = DirectRelAddition(args)
    generator.generate("direct_rel_only")