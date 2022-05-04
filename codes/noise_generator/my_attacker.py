# proposed by "Data Poisoning Attack against Knowledge Graph Embedding"
# we use the Direct Attack in the paper
# we want to find the triple (h', r', t') = argmax(f(h,r',t') - f(h+dh, r', t'))
# CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/my_attacker.py --init_checkpoint ./models/RotatE_FB15k-237_baseline

from direct_addition import *


class MyAddition(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(MyAddition, self).__init__(args)
        self.score_func = lambda s1, s2: args.lambda1 * s1 * 1.0 / args.lambda2 * s2
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
            # embed()
            score = score.detach().cpu().numpy().tolist()
            cand_scores += score
        cand_scores = np.array(cand_scores)
        idx = np.argmax(cand_scores)
        score = cand_scores[idx]
        if mode == "head-batch":
            return (h, cand_r_list[idx], cand_e_list[idx]), score.item()
        return (cand_e_list[idx], cand_r_list[idx], t), score.item()

    def get_noise_triples(self):
        noise_triples = set()
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

if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    generator = MyAddition(args)
    generator.generate("my")