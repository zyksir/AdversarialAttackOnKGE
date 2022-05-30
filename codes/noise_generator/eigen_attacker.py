# follow the code from https://github.com/abojchevski/node_embedding_attack
# paper link: [Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093)
# CUDA_VISIBLE_DEVICES=3 python codes/noise_generator/eigen_attacker.py --init_checkpoint ./models/TransE_FB15k-237_baseline
from random_noise import *
import numba
import scipy.sparse as sp
import scipy.linalg as spl
from rich import print
from rich.progress import track


# @numba.jit(nopython=True)
def sum_of_powers(x, power):
    """For each x_i, computes \sum_{r=1}^{pow) x_i^r (elementwise sum of powers).

    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)

class EigenAttacker(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(EigenAttacker, self).__init__(args)
        self.name = "EigenAttacker"

    def get_relation2adj_matrix(self):
        relation2adj_matrix = {}
        relation2h_t = {}
        for h, r, t in self.input_data.all_true_triples:
            if r not in relation2h_t:
                relation2h_t[r] = set()
            relation2h_t[r].add((h, t))
        for relation in self.all_relations:
            # hint: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            adj_shape = (self.args.nentity, self.args.nentity)
            adj_data = np.ones(len(relation2h_t[relation]))
            row_ind, col_ind = zip(*(list(relation2h_t[relation])))
            row_ind, col_ind = list(row_ind), list(col_ind)
            adj_matrix = sp.csr_matrix((adj_data, (row_ind, col_ind)), shape=adj_shape)
            relation2adj_matrix[relation] = self._standardize(adj_matrix)
        return relation2adj_matrix

    def _standardize(self, adj_matrix):
        """
        Make the graph undirected and select only the nodes
         belonging to the largest connected component.

        :param adj_matrix: sp.spmatrix
            Sparse adjacency matrix

        :return:
            standardized_adj_matrix: sp.spmatrix
                Standardized sparse adjacency matrix.
        """
        standardized_adj_matrix = adj_matrix.copy()
        standardized_adj_matrix[standardized_adj_matrix != 0] = 1                               # unweighted
        standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)    # symmetry

        # select the largest connected component
        _, components = sp.csgraph.connected_components(standardized_adj_matrix)
        c_ids, c_counts = np.unique(components, return_counts=True)
        id_max_component = c_ids[c_counts.argmax()]
        select = np.where(components == id_max_component)[0]
        standardized_adj_matrix = standardized_adj_matrix[select][:, select]

        # remove self-loops
        standardized_adj_matrix = standardized_adj_matrix.tolil()
        standardized_adj_matrix.setdiag(0)
        standardized_adj_matrix = standardized_adj_matrix.tocsr()
        standardized_adj_matrix.eliminate_zeros()

        return standardized_adj_matrix, select

    def get_noise_triples(self):
        noise_triples = set()
        relation2adj_matrix = self.get_relation2adj_matrix()
        dim, window_size = self.args.hidden_dim, 5
        for relation, (adj_matrix, select) in track(relation2adj_matrix.items()):
            print(adj_matrix.shape[0])
            n_flips = sum([r == relation for h, r, t in self.target_triples])
            try:
                candidates = self._generate_candidates_addition(adj_matrix, adj_matrix.shape[0] * 5,
                                                                seed=int(time.time()))
                r_flips = self._perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size)
            except:
                candidate_h = random.choices(self.all_entities, k=n_flips)
                candidate_t = random.choices(self.all_entities, k=n_flips)
                for i in range(n_flips):
                    noise_triples.add((candidate_h[i], relation, candidate_t[i]))
                continue
            for h, t in r_flips:
                noise_triples.add((select[h], relation, select[t]))
        return noise_triples

    def _generate_candidates_addition(self, adj_matrix, n_candidates, seed=0):
        """Generates candidate edge flips for addition (non-edge -> edge).

        adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param n_candidates: int
            Number of candidates to generate.
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        np.random.seed(seed)
        num_nodes = adj_matrix.shape[0]

        candidates = np.random.randint(0, num_nodes, [n_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj_matrix[candidates[:, 0], candidates[:, 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:n_candidates]

        # assert len(candidates) == n_candidates

        return candidates

    def _perturbation_top_flips(self, adj_matrix, candidates, n_flips, dim, window_size):
        """Selects the top (n_flips) number of flips using our perturbation attack.

        :param adj_matrix: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_flips: int
            Number of flips to select
        :param dim: int
            Dimensionality of the embeddings.
        :param window_size: int
            Co-occurence window size.
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        n_nodes = adj_matrix.shape[0]
        # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
        delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1

        # generalized eigenvalues/eigenvectors
        deg_matrix = np.diag(adj_matrix.sum(1).A1)
        vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)

        loss_for_candidates = self._estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes, dim, window_size)
        top_flips = candidates[loss_for_candidates.argsort()[-n_flips:]]

        return top_flips

    # @numba.jit(nopython=True)
    def _estimate_loss_with_delta_eigenvals(self, candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
        """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.

        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips,
        :param flip_indicator: np.ndarray, shape [?]
            Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
        :param vals_org: np.ndarray, shape [n]
            The generalized eigenvalues of the clean graph
        :param vecs_org: np.ndarray, shape [n, n]
            The generalized eigenvectors of the clean graph
        :param n_nodes: int
            Number of nodes
        :param dim: int
            Embedding dimension
        :param window_size: int
            Size of the window
        :return: np.ndarray, shape [?]
            Estimated loss for each candidate flip
        """

        loss_est = np.zeros(len(candidates))
        for x in range(len(candidates)):
            i, j = candidates[x]
            vals_est = vals_org + flip_indicator[x] * (
                    2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))

            vals_sum_powers = sum_of_powers(vals_est, window_size)

            loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))
            loss_est[x] = loss_ij

        return loss_est


class CentralityAttacker(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(CentralityAttacker, self).__init__(args)
        self.name = "CentralityAttacker"

    def get_node2degree(self):
        node2degree = {}
        for h, _, t in self.input_data.all_true_triples:
            if h not in node2degree:
                node2degree[h] = 0
            if t not in node2degree:
                node2degree[t] = 0
            node2degree[h] += 1
            node2degree[t] += 1
        return node2degree

    def get_noise_triples(self):
        node2degree = self.get_node2degree()
        args.num_cand = len(self.target_triples) * 10000
        all_true_triples = set(self.input_data.all_true_triples)
        candidate_h = random.choices(self.all_entities, k=args.num_cand)
        candidate_r = random.choices(self.all_relations, k=args.num_cand)
        candidate_t = random.choices(self.all_entities, k=args.num_cand)
        cand_triple = list(set(zip(candidate_h, candidate_r, candidate_t)).difference(all_true_triples))
        noise_triples = sorted(cand_triple, key=lambda x:(node2degree[x[0]]+node2degree[x[1]], node2degree[x[0]], node2degree[x[1]]))[:len(self.target_triples)]
        return noise_triples


if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    generator = EigenAttacker(args)
    generator.generate("eigen")

    generator = CentralityAttacker(args)
    generator.generate("centrality")