# proposed by "Data Poisoning Attack against Knowledge Graph Embedding"
# we use the Direct Attack in the paper
# we want to find the triple (h', r', t') = argmax(f(h,r',t') - f(h+dh, r', t'))
# CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/least_confidence.py --init_checkpoint ./models/RotatE_FB15k-237_baseline --corruption_factor 70 --num_cand_batch 2048
import heapq
from direct_addition import *

class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []
        self.max_data = None

    def push(self, elem):
        if self.max_data is None:
            self.max_data = elem
        else:
            self.max_data = max(self.max_data, elem)
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return [heapq.heappop(self.data)[1] for _ in range(len(self.data))]

class LeastConfidenceAddition(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(LeastConfidenceAddition, self).__init__(args)
        self.topk_heap = TopKHeap(len(self.target_triples))
        self.name = "least_conf"

    def get_noise_triples(self):
        num_iter = args.corruption_factor
        all_true_triples = set(self.input_data.all_true_triples)
        noise_triples = set()
        for k in range(num_iter):
            args.num_cand = np.math.ceil((args.nentity*args.nrelation)*5 / 100)
            cand_h_list = random.choices(self.all_entities, k=args.num_cand)
            cand_r_list = random.choices(self.all_relations, k=args.num_cand)
            cand_t_list = random.choices(self.all_entities, k=args.num_cand)
            cand_triple = list(zip(cand_h_list, cand_r_list, cand_t_list))
            cand_triple = list(set(cand_triple).difference(all_true_triples | noise_triples))
            noise_triples = noise_triples | set(cand_triple)
            cand_h_list, cand_r_list, cand_t_list = zip(*cand_triple)
            cand_h_list, cand_r_list, cand_t_list = list(cand_h_list), list(cand_r_list), list(cand_t_list)
            train = get_input_data(args).train_triples
            graph = torch.tensor(train, dtype=torch.long).to('cuda:0')
            b_begin, args.num_cand = 0, len(cand_h_list)
            while b_begin < args.num_cand:
                sys.stdout.write("%d/%d in %d/%d\r" % (b_begin, k, args.num_cand, num_iter))
                sys.stdout.flush()
                b_cand_h = cand_h_list[b_begin: b_begin + args.num_cand_batch]
                b_cand_r = cand_r_list[b_begin: b_begin + args.num_cand_batch]
                b_cand_t = cand_t_list[b_begin: b_begin + args.num_cand_batch]
                b_begin += args.num_cand_batch

                embed_cand_h = self.kge_model.node_embeddings[b_cand_h]
                embed_cand_r = self.kge_model.scoring_function.relations[b_cand_r]
                embed_cand_t = self.kge_model.node_embeddings[b_cand_t]
                can_triples = []
                for i in range(len(b_cand_h)):
                  can_triples.append([b_cand_h[i], b_cand_r[i], b_cand_t[i]])
                can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias

                s, _, _ = self.kge_model(graph, can_triples, x)
                
                s = s.view(-1).detach().cpu().numpy().tolist()
                for i in range(len(b_cand_h)):
                    self.topk_heap.push((-s[i], (b_cand_h[i], b_cand_r[i], b_cand_t[i])))
        # embed()
        sys.stdout.write("max score in generated triples is %f, min score is %f\n" % (-self.topk_heap.data[0][0], -self.topk_heap.max_data[0]))
        return self.topk_heap.topk()


class LeastScoreAddition(DirectAddition):
    def __init__(self, args):
        super(LeastScoreAddition, self).__init__(args)
        self.name = "least_score"

    def get_noise_for_head(self, test_triple, mode="head-batch"):
        args = self.args
        h, r, t = test_triple

        cand_e_list = self.all_entities
        args.num_cand = len(cand_e_list)

        embed_h = self.kge_model.node_embeddings[h]
        embed_r = self.kge_model.scoring_function.relations[r]
        embed_t = self.kge_model.node_embeddings[t]
        train = get_input_data(args).train_triples
        graph = torch.tensor(train, dtype=torch.long).to('cuda:0')
        b_begin = 0
        cand_scores = []
        with torch.no_grad():
            while b_begin < args.num_cand:
                b_cand_e = cand_e_list[b_begin: b_begin + args.num_cand]
                b_begin += args.num_cand

                if mode == "head-batch":
                    can_triples = []
                    for i in range(args.num_cand):
                      can_triples.append([h, r, b_cand_e[i]])
                    can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                    x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
                    
                    score, _, _ = self.kge_model(graph, can_triples, x)
                    embed_cand_t = self.kge_model.node_embeddings[b_cand_e]
                else:
                    embed_cand_h = self.kge_model.node_embeddings[b_cand_e]
                    can_triples = []
                    for i in range(args.num_cand):
                      can_triples.append([b_cand_e[i], r, t])
                    can_triples = torch.tensor(can_triples, dtype=torch.long).to('cuda:0')
                    x = self.kge_model.node_embeddings + self.kge_model.node_embeddings_bias
                    
                    score, _, _ = self.kge_model(graph, can_triples, x)
                score = score.detach().cpu().numpy().tolist()
                cand_scores += score
        cand_scores = np.array(cand_scores)
        idx = np.argmax(cand_scores)
        score = cand_scores[idx]
        if mode == "head-batch":
            return (h, r, cand_e_list[idx]), score.item()
        return (cand_e_list[idx], r, t), score.item()


if __name__ == "__main__":
    args = get_noise_args()
    #override_config(args)
    generator = LeastConfidenceAddition(args)
    generator.generate("least_conf_%d" % args.corruption_factor)

    generator = LeastScoreAddition(args)
    generator.generate("least_score")