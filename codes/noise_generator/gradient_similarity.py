"""
CUDA_VISIBLE_DEVICES=0 python codes/noise_generator/gradient_similarity.py \
                              --init_checkpoint ./models/RotatE_FB15k-237_baseline
"""
import time
import torch

from instance_similarity import *
from model import KGEModel
import torch.autograd as autograd
from dataloader import *


def get_non_zero_idx(matrix1, matrix2):
    idx1 = set(torch.nonzero(matrix1)[:, 0].detach().cpu().numpy().tolist())
    idx2 = set(torch.nonzero(matrix2)[:, 0].detach().cpu().numpy().tolist())
    idx = list(idx1.intersection(idx2))
    return idx

def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=False) -> torch.Tensor:
    """ refer:https://zhuanlan.zhihu.com/p/530879775
    基于 torch.autograd.grad 函数的更清晰明了的 API，功能是计算一个雅可比矩阵。

    Args:
        y (torch.Tensor): 函数输出向量
        x (torch.Tensor): 函数输入向量
        need_higher_grad (bool, optional): 是否需要计算高阶导数，如果确定不需要可以设置为 False 以节约资源. 默认为 True.

    Returns:
        torch.Tensor: 计算好的“雅可比矩阵”。注意！输出的“雅可比矩阵”形状为 y.shape + x.shape。例如：y 是 n 个元素的张量，y.shape = [n]；x 是 m 个元素的张量，x.shape = [m]，则输出的雅可比矩阵形状为 n x m，符合常见的数学定义。
        但是若 y 是 1 x n 的张量，y.shape = [1,n]；x 是 1 x m 的张量，x.shape = [1,m]，则输出的雅可比矩阵形状为1 x n x 1 x m，如果嫌弃多余的维度可以自行使用 torch.squeeze(Jac) 一步到位。
        这样设计是因为考虑到 y 是 n1 x n2 的张量； 是 m1 x m2 的张量（或者形状更复杂的张量）时，输出 n1 x n2 x m1 x m2 （或对应更复杂形状）更有直观含义，方便用户知道哪一个元素对应的是哪一个偏导。
    """
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y)).cuda(),),
        retain_graph=True,
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac.reshape(shape=(y.shape + x.shape))
    return Jac



class InstanceAttributionCosGrad(InstanceAttributionCos):
    def __init__(self, args):
        super(InstanceAttributionCosGrad, self).__init__(args)
        self.similarity_func = lambda grad, nghbr_grad: F.cosine_similarity(grad, nghbr_grad)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        self.name = "gs_cos"

        named_parameters = list(self.kge_model.named_parameters())
        self.param_list = []
        for n, p in named_parameters:
            if p.requires_grad:
                self.param_list.append(p)

    def get_loss(self, triple):
        h, r, t = triple[:, 0], triple[:, 1], triple[:, 2]
        embed_h = self.kge_model.entity_embedding[h]
        embed_r = self.kge_model.relation_embedding[r]
        embed_t = self.kge_model.entity_embedding[t]
        loss = 0
        pred_h_embedding = self.kge_model.predict_embedding(embed_t, embed_r, "head-mode")
        pred_h = torch.mm(pred_h_embedding.squeeze(1), self.kge_model.entity_embedding.transpose(1,0))
        pred_h = torch.sigmoid(pred_h)
        loss += self.loss_func(pred_h, h)

        pred_t_embedding = self.kge_model.predict_embedding(embed_h, embed_r, "tail-mode")
        pred_t = torch.mm(pred_t_embedding.squeeze(1), self.kge_model.entity_embedding.transpose(1,0))
        pred_t = torch.sigmoid(pred_t)
        loss += self.loss_func(pred_t, t)
        return loss

    # given loss of two triples, calculate the score between them
    def calc_influential_score(self, loss1, loss2):
        batch_size = loss2.view(-1).shape[0]
        grad_e1 = jacobian(loss1, self.param_list[0]) # size:1*E*dim
        grad_r1 = jacobian(loss1, self.param_list[1]) # size:1*R*dim
        # embed()
        grad_e2 = jacobian(loss2, self.param_list[0]) # size:B*E*dim
        grad_r2 = jacobian(loss2, self.param_list[1]) # size:B*R*dim
        # grad_e1, grad_r1 = autograd.grad(loss1, self.param_list, retain_graph=True)
        # grad_e2, grad_r2 = autograd.grad(sumed_loss2, batched_param_list, retain_graph=True, \
        #     grad_outputs=(torch.eye(torch.numel(sumed_loss2)),), is_grads_batched=True)
        # embed()
        grad_e1, grad_e2 = grad_e1.view(1, -1), grad_e2.view(batch_size, -1)
        grad_r1, grad_r2 = grad_r1.view(1, -1), grad_r2.view(batch_size, -1)
        score = self.similarity_func(grad_e1, grad_e2) + self.similarity_func(grad_r1, grad_r2)
        score = score.detach().cpu().numpy().tolist()
        del grad_e1
        del grad_e2
        del grad_r1
        del grad_r2
        torch.cuda.empty_cache()
        return score

    def get_influential_triples(self):
        args = self.args
        influential_triples_path = os.path.join(args.init_checkpoint, "%s_influential_triples.pkl" % self.name)
        if not args.no_store and os.path.exists(influential_triples_path):
            with open(influential_triples_path, "rb") as f:
                triple2influential_triple = pickle.load(f)
                if (triple2influential_triple is not None and type(triple2influential_triple) == type({1:1}) and all([triple in triple2influential_triple for triple in self.target_triples])):
                    return triple2influential_triple
        
        triple2influential_triple = {}
        for i, target_triple in enumerate(self.target_triples):
            ngbhrs = self.triple2nghbrs[target_triple]
            target_triple = torch.LongTensor(target_triple).view(-1, 3).cuda()
            target_loss = self.get_loss(target_triple)
            if len(ngbhrs) == 0:
                print(f"we don't need to attack {target_triple} in {args.data_path}")
                continue
            nghbr_sim = []
            b_beign = 0
            # for i, b_ngbhrs in enumerate(ngbhrs):
            while b_beign < len(ngbhrs):
                b_ngbhrs = ngbhrs[b_beign: b_beign+args.num_cand_batch]
                b_beign += args.num_cand_batch

                t1 = time.time()
                b_ngbhrs = torch.LongTensor(b_ngbhrs).view(-1, 3).cuda()
                ngbhr_loss = self.get_loss(b_ngbhrs)
                grad_sim = self.calc_influential_score(target_loss, ngbhr_loss)
                nghbr_sim += grad_sim
                t2 = time.time()
                # print(f"time used: {t2 - t1}: {b_beign}/{len(ngbhrs)}")
            nghbr_sim = np.array(nghbr_sim)
            idx = np.argmax(nghbr_sim)
            target_triple = tuple(target_triple.view(-1).detach().cpu().tolist())
            triple2influential_triple[target_triple] = ngbhrs[idx]

            sys.stdout.write("influential:\t%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
        if not args.no_store:
            with open(influential_triples_path, "wb") as fw:
                pickle.dump(triple2influential_triple, fw)
        return triple2influential_triple


class InstanceAttributionDotGrad(InstanceAttributionCosGrad):
    def __init__(self, args):
        super(InstanceAttributionDotGrad, self).__init__(args)
        self.similarity_func = lambda grad, nghbr_grad: torch.matmul(grad, nghbr_grad.T)
        self.name = "gs_dot"
        # embed()

class InstanceAttributionL2Grad(InstanceAttributionCosGrad):
    def __init__(self, args):
        super(InstanceAttributionL2Grad, self).__init__(args)
        self.similarity_func = lambda grad, nghbr_grad: -torch.norm((grad-nghbr_grad), p=2, dim=-1)
        self.name = "gs_l2"


if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    # print(f"after override_config, args={args.__dict__}")
    generator = InstanceAttributionCosGrad(args)
    generator.generate("gs_cos")

    generator = InstanceAttributionDotGrad(args)
    generator.generate("gs_dot")

    generator = InstanceAttributionL2Grad(args)
    generator.generate("gs_l2")