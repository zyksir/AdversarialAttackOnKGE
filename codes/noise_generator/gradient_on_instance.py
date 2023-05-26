# proposed by "Data Poisoning Attack against Knowledge Graph Embedding"
# we use the Direct Attack in the paper
# we want to find the triple (h', r', t') = argmax(f(h,r',t') - f(h+dh, r', t'))
# CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/gradient_on_instance.py --init_checkpoint ./models/RotatE_FB15k-237_baseline
from collections import defaultdict
from gradient_similarity import *
from torch.nn import functional as F

def direct_addition_add_true_triple(self, triple):
    h, r, t = triple
    self.true_rel_tail[h].add((r, t))
    self.true_rel_head[t].add((r, h))

def direct_addition_init(generator):
    self = generator
    args = self.args
    args.num_cand = np.math.ceil((args.nentity*args.nrelation)*args.corruption_factor / 100)
    self.score_func = lambda s1, s2: args.lambda1 * s1 - args.lambda2 * s2
    self.true_rel_head, self.true_rel_tail = defaultdict(set), defaultdict(set)
    for triple in self.input_data.all_true_triples:
        direct_addition_add_true_triple(self, triple)

def direct_addition_get_noise_for_head(generator, test_triple, mode="head-batch"):
    self = generator
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

def new_get_noise_triples(generator):
    self = generator
    noise_triples = self.noise_triples
    influential_triples = self.get_influential_triples()
    for i in range(len(self.target_triples)):
        sys.stdout.write("%d in %d\r" % (i, len(self.target_triples)))
        sys.stdout.flush()
        h, r, t = self.target_triples[i]
        if (h, r, t) not in influential_triples:
            continue
        hi, ri, ti = influential_triples[(h, r, t)]
        if ti in [h, t]:
            mode = "tail-batch"
        elif hi in [h, t]:
            mode = "head-batch"
        else:
            print("unexpected behavior")
        noise_triple = direct_addition_get_noise_for_head(self, (hi, ri, ti), mode=mode)[0]
        noise_triples.add(noise_triple)
        
    return list(noise_triples)

class GradientInstanceIsCos(InstanceAttributionCos):
    def __init__(self, args):
        super(GradientInstanceIsCos, self).__init__(args)
        direct_addition_init(self)

    def get_noise_triples(self):
        return new_get_noise_triples(self)

class GradientInstanceIsDot(InstanceAttributionDot):
    def __init__(self, args):
        super(GradientInstanceIsDot, self).__init__(args)
        direct_addition_init(self)

    def get_noise_triples(self):
        return new_get_noise_triples(self)


class GradientInstanceIsL2(InstanceAttributionL2):
    def __init__(self, args):
        super(GradientInstanceIsL2, self).__init__(args)
        direct_addition_init(self)
    
    def get_noise_triples(self):
        return new_get_noise_triples(self)

class GradientInstanceGsCos(InstanceAttributionCosGrad):
    def __init__(self, args):
        super(GradientInstanceGsCos, self).__init__(args)
        direct_addition_init(self)

    def get_noise_triples(self):
        return new_get_noise_triples(self)

class GradientInstanceGsDot(InstanceAttributionDotGrad):
    def __init__(self, args):
        super(GradientInstanceGsDot, self).__init__(args)
        direct_addition_init(self)

    def get_noise_triples(self):
        return new_get_noise_triples(self)


class GradientInstanceGsL2(InstanceAttributionL2Grad):
    def __init__(self, args):
        super(GradientInstanceGsL2, self).__init__(args)
        direct_addition_init(self)
    
    def get_noise_triples(self):
        return new_get_noise_triples(self)


if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    generator = GradientInstanceIsCos(args)
    generator.generate("grad_is_cos")

    generator = GradientInstanceIsDot(args)
    generator.generate("grad_is_dot")

    generator = GradientInstanceIsL2(args)
    generator.generate("grad_is_l2")

    generator = GradientInstanceGsCos(args)
    generator.generate("grad_gs_cos")

    generator = GradientInstanceGsDot(args)
    generator.generate("grad_gs_dot")

    generator = GradientInstanceGsL2(args)
    generator.generate("grad_gs_l2")
