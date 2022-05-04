"""
DON't RUN this py file, it is too slow.
CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/grad_instance_attribution.py \
                              --init_checkpoint ./models/RotatE_FB15k-237_baseline
"""
import torch

from instance_attribution import *
from model import KGEModel
import torch.autograd as autograd
from dataloader import *


def get_non_zero_idx(matrix1, matrix2):
    idx1 = set(torch.nonzero(matrix1)[:, 0].detach().cpu().numpy().tolist())
    idx2 = set(torch.nonzero(matrix2)[:, 0].detach().cpu().numpy().tolist())
    idx = list(idx1.intersection(idx2))
    return idx


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

        self.triple2loss = {}
        # train_dataloader_head = DataLoader(
        #     TrainDataset(self.trainer.input_data.train_triples + self.target_triples,
        #                  args.nentity, args.nrelation,
        #                  args.negative_sample_size, 'head-batch'),
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=max(1, args.cpu_num // 2),
        #     collate_fn=TrainDataset.collate_fn
        # )

        # train_dataloader_tail = DataLoader(
        #     TrainDataset(self.trainer.input_data.train_triples + self.target_triples,
        #                  args.nentity, args.nrelation,
        #                  args.negative_sample_size, 'tail-batch'),
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=max(1, args.cpu_num // 2),
        #     collate_fn=TrainDataset.collate_fn
        # )
        # for positive_sample, negative_sample, _, mode in tqdm(train_dataloader_head):
        #     if args.cuda:
        #         positive_sample = positive_sample.cuda()
        #         negative_sample = negative_sample.cuda()
        #     positive_score, negative_score = KGEModel.compute_score(
        #         self.kge_model, args, positive_sample, negative_sample, mode)
        #     for i, (h, r, t) in enumerate(positive_sample.detach().cpu().numpy().tolist()):
        #         self.triple2loss[(h, r, t)] = (-positive_score[i]-negative_score[i])/2
        # for positive_sample, negative_sample, _, mode in tqdm(train_dataloader_tail):
        #     if args.cuda:
        #         positive_sample = positive_sample.cuda()
        #         negative_sample = negative_sample.cuda()
        #     positive_score, negative_score = KGEModel.compute_score(
        #         self.kge_model, args, positive_sample, negative_sample, mode)
        #     for i, (h, r, t) in enumerate(positive_sample.detach().cpu().numpy().tolist()):
        #         self.triple2loss[(h, r, t)] = (-positive_score[i]-negative_score[i])/4 + self.triple2loss[(h, r, t)]/2
        # # embed()


    def get_mode_loss(self, triple, mode):
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
        loss, _, _ = KGEModel.compute_loss(self.kge_model, self.args, positive_sample, negative_sample, subsampling_weight, mode)
        return loss

    def get_loss(self, triple):
        if triple in self.triple2loss:
            return self.triple2loss[triple]
        loss1 = self.get_mode_loss(triple, "head-mode")
        loss2 = self.get_mode_loss(triple, "tail-mode")
        loss = (loss1 + loss2) / 2
        self.triple2loss[triple] = loss
        return loss

    # given loss of two triples, calculate the score between them
    def calc_influential_score(self, loss1, loss2):
        grad_e1, grad_r1 = autograd.grad(loss1, self.param_list, retain_graph=True)
        grad_e2, grad_r2 = autograd.grad(loss2, self.param_list, retain_graph=True)
        idx = get_non_zero_idx(grad_e1, grad_e2)
        grad_e1, grad_e2 = grad_e1[idx].view(-1), grad_e2[idx].view(-1)
        idx = get_non_zero_idx(grad_r1, grad_r2)
        grad_r1, grad_r2 = grad_r1[idx].view(-1), grad_r2[idx].view(-1)
        grad1 = torch.cat([grad_e1, grad_r1], 0)
        grad2 = torch.cat([grad_e2, grad_r2], 0)
        score = F.cosine_similarity(grad1, grad2, dim=0)
        return score

    def get_influential_triples(self):
        args = self.args
        influential_triples_path = os.path.join(args.init_checkpoint, "%s_influential_triples.pkl" % self.name)
        if os.path.exists(influential_triples_path):
            with open(influential_triples_path, "rb") as f:
                return pickle.load(f)

        triple2influential_triple = {}
        for i, target_triple in enumerate(self.target_triples):
            sys.stdout.write("influential:\t%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            target_loss = self.get_loss(target_triple)
            ngbhrs = self.triple2nghbrs[target_triple]
            if len(ngbhrs) == 0:
                print(f"we don't need to attack {target_triple} in {args.data_path}")
                continue
            nghbr_sim = []
            for ngbhr_triple in ngbhrs:
                ngbhr_loss = self.get_loss(ngbhr_triple)
                grad_sim = self.calc_influential_score(target_loss, ngbhr_loss)
                nghbr_sim.append(grad_sim)
            nghbr_sim = np.array(nghbr_sim)
            idx = np.argmax(nghbr_sim)
            triple2influential_triple[target_triple] = ngbhrs[idx]

        with open(influential_triples_path, "wb") as fw:
            pickle.dump(triple2influential_triple, fw)
        return triple2influential_triple


class InstanceAttributionDotGrad(InstanceAttributionCosGrad):
    def __init__(self, args):
        super(InstanceAttributionCosGrad, self).__init__(args)
        self.similary_func = lambda grad, nghbr_grad: torch.matmul(grad, nghbr_grad)
        self.name = "dot_grad"

class InstanceAttributionL2Grad(InstanceAttributionCosGrad):
    def __init__(self, args):
        super(InstanceAttributionL2Grad, self).__init__(args)
        self.similary_func = lambda grad, nghbr_grad: -torch.norm((grad-nghbr_grad), p=2, dim=-1)
        self.name = "l2_grad"


# if __name__ == "__main__":
#     args = get_noise_args()
#     override_config(args)
#     print(f"after override_config, args={args.__dict__}")
#     generator = InstanceAttributionCosGrad(args)
#     generator.generate("if_cos_grad")
#
#     generator = InstanceAttributionDotGrad(args)
#     generator.generate("if_dot_grad")
#
#     generator = InstanceAttributionL2Grad(args)
#     generator.generate("if_l2_grad")