# follow the code from https://github.com/abojchevski/node_embedding_attack
# paper link: [Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093)
from random_noise import *
import numba
import scipy.sparse as sp
import scipy.linalg as spl


@numba.jit(nopython=True)
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
        for relation in self.all_relations:
            # TODO: set values
            # hint: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            adj_data, adj_indices, adj_indptr, adj_shape = None, None, None, None

            adj_matrix = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
            relation2adj_matrix[relation] = adj_matrix
        return relation2adj_matrix


    def get_noise_triples(self):
        relation2adj_matrix = self.get_relation2adj_matrix()
        for relation, adj_matrix in relation2adj_matrix.items():
            candidates = None                               # TODO: generate candidates for every relation
            n_flips, dim, window_size = None, None, None    # TODO: set values
            our_flips = self._perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size)

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

    @numba.jit(nopython=True)
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

