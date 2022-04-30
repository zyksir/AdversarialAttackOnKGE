"""
CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/grad_instance_attribution.py \
                              --init_checkpoint ./models/RotatE_FB15k-237_baseline
"""
from instance_attribution import *
from model import KGEModel
import torch.autograd as autograd

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

class InstanceAttributionCosGrad(InstanceAttributionCos):
    def __init__(self, args):
        super(InstanceAttributionCosGrad, self).__init__(args)
        self.similary_func = lambda grad, nghbr_grad: F.cosine_similarity(grad, nghbr_grad, dim=0)
        self.name = "cos_grad"

        named_parameters = list(self.kge_model.named_parameters())
        self.param_list = []
        for n, p in named_parameters:
            if p.requires_grad:
                self.param_list.append(p)

    def get_loss(self, triple, mode):
        if mode == "head-mode":
            dataset = self.trainer.train_dataloader_head.dataset
        else:
            dataset = self.trainer.train_dataloader_tail.dataset
        positive_sample, negative_sample, subsampling_weight, mode = dataset.get_negative_sample(triple, if_reweight=False)
        positive_sample, negative_sample, subsampling_weight = \
            positive_sample.unsqueeze(dim=0), negative_sample.unsqueeze(dim=0), subsampling_weight.unsqueeze(dim=0)
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        loss = KGEModel.compute_loss(self.kge_model, self.args, positive_sample, negative_sample, subsampling_weight, mode)
        return loss

    def get_grad(self, triple):
        loss1 = self.get_loss(triple, "head-mode")
        loss2 = self.get_loss(triple, "tail-mode")
        loss = (loss1 + loss2) / 2
        grads = autograd.grad(loss, self.param_list)     # calc_grad
        return gather_flat_grad(grads)

    def get_influential_triples(self):
        influential_triples_path = os.path.join(args.init_checkpoint, "%s_influential_triples.pkl" % self.name)
        if os.path.exists(influential_triples_path):
            with open(influential_triples_path, "rb") as f:
                return pickle.load(f)

        triple2influential_triple = {}
        for i, (h, r, t) in enumerate(self.target_triples):
            sys.stdout.write("influential:\t%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            target_grad = self.get_grad((h, r, t))
            ngbhrs = self.triple2nghbrs[(h, r, t)]
            if len(ngbhrs) == 0:
                print(f"we don't need to attack {h, r, t} in {args.data_path}")
                continue
            nghbr_sim = []
            for ngbhr_triple in ngbhrs:
                ngbhr_grad = self.get_grad(ngbhr_triple)
                grad_sim = self.similary_func(target_grad, ngbhr_grad).item()
                nghbr_sim.append(grad_sim)
            nghbr_sim = np.array(nghbr_sim)
            idx = np.argmax(nghbr_sim)
            triple2influential_triple[(h, r, t)] = ngbhrs[idx]

        with open(influential_triples_path, "wb") as fw:
            pickle.dump(triple2influential_triple, fw)
        return triple2influential_triple


if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    print(f"after override_config, args={args.__dict__}")
    generator = InstanceAttributionCosGrad(args)
    generator.generate("if_cos_grad")