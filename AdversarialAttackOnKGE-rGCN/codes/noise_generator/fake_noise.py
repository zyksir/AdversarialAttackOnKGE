"""
CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/fake_noise.py \
                              --init_checkpoint ./models/RotatE_FB15k-237_baseline
"""
from instance_attribution import *


class FakeAttackCos(InstanceAttributionCos):
    def __init__(self, args):
        super(FakeAttackCos, self).__init__(args)
        self.similary_func = lambda emb, ent_emb: F.cosine_similarity(emb, ent_emb)
        self.name = "fake_cos"

    def get_influential_triples(self):
        triple2influential_triple = {}
        for i, target_triple in enumerate(self.target_triples):
            triple2influential_triple[target_triple] = target_triple
        return triple2influential_triple

    def find_least_similary_entity(self, entity, r, e, mode):
        train_triples = self.train_triples
        ent_embed = self.kge_model.entity_embedding[entity].view(1, -1)
        cos_sim_ent = self.similary_func(ent_embed, self.entity_embedding)
        filter_ent = None
        if mode == "head-mode":
            filter_ent = train_triples[np.where((train_triples[:, 2] == e) & (train_triples[:, 1] == r)), 0]
        elif mode == "tail-mode":
            filter_ent = train_triples[np.where((train_triples[:, 0] == e) & (train_triples[:, 1] == r)), 2]
        cos_sim_ent[filter_ent.squeeze()] = 1e8
        idx = torch.argmin(cos_sim_ent).item()
        return idx

class FakeAttackDot(FakeAttackCos):
    def __init__(self, args):
        super(FakeAttackDot, self).__init__(args)
        self.similary_func = lambda vec, nghbr_vec: torch.matmul(vec, nghbr_vec.t()).view(-1)
        self.name = "fake_dot"


class FakeAttackL2(FakeAttackCos):
    def __init__(self, args):
        super(FakeAttackL2, self).__init__(args)
        self.similary_func = lambda vec, nghbr_vec: -torch.norm((nghbr_vec-vec), p=2, dim=-1)
        self.name = "fake_l2"


if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    print(f"after override_config, args={args.__dict__}")
    generator = FakeAttackCos(args)
    generator.generate("fake_cos")

    generator = FakeAttackDot(args)
    generator.generate("fake_dot")

    generator = FakeAttackL2(args)
    generator.generate("fake_l2")