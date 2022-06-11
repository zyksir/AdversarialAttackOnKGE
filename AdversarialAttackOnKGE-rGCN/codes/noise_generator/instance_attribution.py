# proposed by "Data Poisoning Attack against Knowledge Graph Embedding"
# we use the Direct Attack in the paper
# we want to find the triple (h', r', t') = argmax(f(h,r',t') - f(h+dh, r', t'))
# CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/instance_attribution.py --init_checkpoint ./models/RotatE_FB15k-237_baseline
from random_noise import *
from torch.nn import functional as F


def generate_nghbrs(target_triples, train_triples):
    '''
    For every triple in target_triples set,
    return the index of neighbouring triple in train triples,
    '''
    triple2nghbrs = {}
    train_triples = np.array(train_triples)
    for h, r, t in target_triples:
        mask = (np.isin(train_triples[:, 0], [h, t]) | np.isin(train_triples[:, 2], [h, t]))
        mask_idx = np.where(mask)[0]
        triple2nghbrs[(h, r, t)] = [tuple(triple) for triple in train_triples[mask_idx].tolist()]
    return triple2nghbrs


class InstanceAttributionCos(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(InstanceAttributionCos, self).__init__(args)
        self.entity_embedding = self.kge_model.node_embeddings
        self.train_triples = np.array(self.input_data.train_triples)
        triple2nghbrs_path = os.path.join(args.data_path, "triple2nghbrs.pkl")
        if not os.path.exists(triple2nghbrs_path):
            with open(triple2nghbrs_path, "wb") as fw:
                self.triple2nghbrs = generate_nghbrs(self.target_triples, self.train_triples)
                pickle.dump(self.triple2nghbrs, fw)
        else:
            with open(triple2nghbrs_path, "rb") as f:
                self.triple2nghbrs = pickle.load(f)
        print(f"generate_nghbrs done")
        self.similary_func = lambda vec, nghbr_vec: F.cosine_similarity(vec, nghbr_vec)
        self.name = "cos"

    def get_influential_triples(self):
        influential_triples_path = os.path.join(args.init_checkpoint, "%s_influential_triples.pkl" % self.name)
        if os.path.exists(influential_triples_path):
            with open(influential_triples_path, "rb") as f:
                return pickle.load(f)
        triple2influential_triple = {}
        for i, (h, r, t) in enumerate(self.target_triples):
            sys.stdout.write("influential:\t%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            sample = torch.LongTensor([h, r, t]).view(1, -1)
            if self.args.cuda:
                sample = sample.cuda()
            x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
            train = get_input_data(args).train_triples
            graph = torch.tensor(train, dtype=torch.long).to('cuda:0')
            vec_score, _, _ = self.kge_model(graph, sample, x)
            vec_score = vec_score.view(1, -1)
            ngbhrs = self.triple2nghbrs[(h, r, t)]
            if len(ngbhrs) == 0:
                print(f"we don't need to attack {h, r, t} in {args.data_path}")
                continue
            b_beign = 0
            nghbr_sim = []
            while b_beign < len(ngbhrs):
                b_ngbhrs = ngbhrs[b_beign: b_beign+args.num_cand_batch]
                b_beign += args.num_cand_batch
                b_ngbhrs = torch.LongTensor(b_ngbhrs).view(-1, 3)
                if self.args.cuda:
                    b_ngbhrs = b_ngbhrs.cuda()
                b_ngbhrs_vec, _, _ = self.kge_model(graph, b_ngbhrs, x)
                b_ngbhrs_vec = b_ngbhrs_vec.view(-1, vec_score.shape[-1])
                b_sim = self.similary_func(vec_score, b_ngbhrs_vec).detach().cpu().numpy().tolist()
                nghbr_sim += b_sim
            nghbr_sim = np.array(nghbr_sim)
            idx = np.argmax(nghbr_sim)
            triple2influential_triple[(h, r, t)] = ngbhrs[idx]
        with open(influential_triples_path, "wb") as fw:
            pickle.dump(triple2influential_triple, fw)
        return triple2influential_triple

    def find_least_similary_entity(self, entity, r, e, mode):
        train_triples = self.train_triples
        ent_embed = self.kge_model.node_embeddings[entity].view(1, -1)
        cos_sim_ent = F.cosine_similarity(ent_embed, self.entity_embedding)
        filter_ent = None
        if mode == "head-mode":
            filter_ent = train_triples[np.where((train_triples[:, 2] == e) & (train_triples[:, 1] == r)), 0]
        elif mode == "tail-mode":
            filter_ent = train_triples[np.where((train_triples[:, 0] == e) & (train_triples[:, 1] == r)), 2]
        cos_sim_ent[filter_ent.squeeze()] = 1e8
        idx = torch.argmin(cos_sim_ent).item()
        return idx

    def get_noise_triples(self):
        noise_triples = set()
        influential_triples = self.get_influential_triples()
        for i in range(len(self.target_triples)):
            sys.stdout.write("%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            h, r, t = self.target_triples[i]
            if (h, r, t) not in influential_triples:
                continue
            hi, ri, ti = influential_triples[(h, r, t)]
            if ti in [h, t]:
                fake_head = self.find_least_similary_entity(hi, ri, ti, mode="head-mode")
                noise_triples.add((fake_head, ri, ti))
            elif hi in [h, t]:
                fake_tail = self.find_least_similary_entity(ti, ri, hi, mode="tail-mode")
                noise_triples.add((hi, ri, fake_tail))
            else:
                print("unexpected behavior")

        return list(noise_triples)


class InstanceAttributionDot(InstanceAttributionCos):
    def __init__(self, args):
        super(InstanceAttributionDot, self).__init__(args)
        self.similary_func = lambda vec, nghbr_vec: torch.matmul(vec, nghbr_vec.t())
        self.name = "dot"


class InstanceAttributionL2(InstanceAttributionCos):
    def __init__(self, args):
        super(InstanceAttributionL2, self).__init__(args)
        self.similary_func = lambda vec, nghbr_vec: -torch.norm((nghbr_vec-vec), p=2, dim=-1)
        self.name = "l2"


if __name__ == "__main__":
    args = get_noise_args()
    #override_config(args)
    print(f"after override_config, args={args.__dict__}")
    generator = InstanceAttributionCos(args)
    generator.generate("if_cos")

    generator = InstanceAttributionDot(args)
    generator.generate("if_dot")

    generator = InstanceAttributionL2(args)
    generator.generate("if_l2")
