import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize

from src.adapters.barycentric import apply_barycentric


def sym_inv_sqrt(M):
    v, Q = np.linalg.eig(M)
    return Q @ np.diag((v + 1e-12) ** (-0.5)) @ Q.T


class SCOTv2(object):

    def __init__(self, data):
        assert type(data) == list and len(data) >= 2, (
            "SCOTv2 requires a list of at least two numpy arrays (cells × features)."
        )
        self.data = data
        self.marginals = []
        self.graphs = []
        self.graphDists = []
        self.couplings = []
        self.flags = []

    def init_marginals(self):
        for i in range(len(self.data)):
            num_cells = self.data[i].shape[0]
            self.marginals.append(torch.ones(num_cells) / num_cells)
        return self.marginals

    def normalize_data(self, norm="l2", bySample=True):
        assert norm in ["l1", "l2", "max", "zscore"], (
            "norm must be one of 'l1', 'l2', 'max', 'zscore'."
        )
        for i in range(len(self.data)):
            if norm == "zscore":
                self.data[i] = StandardScaler().fit_transform(self.data[i])
            else:
                axis = 1 if (bySample is True or bySample is None) else 0
                self.data[i] = normalize(self.data[i], norm=norm, axis=axis)
        return self.data

    def construct_graph(self, k=20, mode="connectivity", metric="correlation"):
        assert mode in ["connectivity", "distance"]
        include_self = mode == "connectivity"
        for i in range(len(self.data)):
            self.graphs.append(
                kneighbors_graph(self.data[i], n_neighbors=k, mode=mode,
                                 metric=metric, include_self=include_self)
            )
        return self.graphs

    def init_graph_distances(self):
        for i in range(len(self.data)):
            shortest = dijkstra(csgraph=csr_matrix(self.graphs[i]),
                                directed=False, return_predecessors=False)
            max_dist = np.nanmax(shortest[shortest != np.inf])
            shortest[shortest > max_dist] = max_dist
            self.graphDists.append(shortest / shortest.max())
        return self.graphDists

    def exp_sinkhorn_solver(self, ecost, u, v, a, b, mass, eps, rho, rho2,
                            nits_sinkhorn, tol_sinkhorn):
        if u is None or v is None:
            u, v = torch.ones_like(a), torch.ones_like(b)
        k = (a * u ** (-eps / rho)).sum() + (b * v ** (-eps / rho)).sum()
        k = k / (2 * (u[:, None] * v[None, :] * ecost * a[:, None] * b[None, :]).sum())
        z = (0.5 * mass * eps) / (2.0 + 0.5 * (eps / rho) + 0.5 * (eps / rho2))
        k = k ** z
        u, v = u * k, v * k
        for _ in range(nits_sinkhorn):
            u_prev = u.clone()
            v = torch.einsum("ij,i->j", ecost, a * u) ** (-1.0 / (1.0 + eps / rho))
            u = torch.einsum("ij,j->i", ecost, b * v) ** (-1.0 / (1.0 + eps / rho2))
            if (u.log() - u_prev.log()).abs().max().item() * eps < tol_sinkhorn:
                break
        pi = u[:, None] * v[None, :] * ecost * a[:, None] * b[None, :]
        return u, v, pi

    def exp_unbalanced_gw(self, a, dx, b, dy, eps=0.01, rho=1.0, rho2=None,
                          nits_plan=3000, tol_plan=1e-6,
                          nits_sinkhorn=3000, tol_sinkhorn=1e-6):
        if rho2 is None:
            rho2 = rho
        pi = a[:, None] * b[None, :] / (a.sum() * b.sum()).sqrt()
        pi_prev = torch.zeros_like(pi)
        up, vp = None, None
        for i in range(nits_plan):
            pi_prev = pi.clone()
            mp = pi.sum()
            distxy = torch.einsum("ij,kj->ik", dx, torch.einsum("kl,jl->kj", dy, pi))
            kl_pi = torch.sum(pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log())
            mu, nu = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
            distxx = torch.einsum("ij,j->i", dx ** 2, mu)
            distyy = torch.einsum("kl,l->k", dy ** 2, nu)
            lcost = (distxx[:, None] + distyy[None, :] - 2 * distxy) + eps * kl_pi
            if rho < float("Inf"):
                lcost = lcost + rho * torch.sum(mu * (mu / a + 1e-10).log())
            if rho2 < float("Inf"):
                lcost = lcost + rho2 * torch.sum(nu * (nu / b + 1e-10).log())
            ecost = (-lcost / (mp * eps)).exp()
            if i % 10 == 0:
                print(f"Unbalanced GW step: {i}")
            up, vp, pi = self.exp_sinkhorn_solver(
                ecost, up, vp, a, b, mp, eps, rho, rho2, nits_sinkhorn, tol_sinkhorn
            )
            flag = not torch.any(torch.isnan(pi)).item()
            pi = (mp / pi.sum()).sqrt() * pi
            if (pi - pi_prev).abs().max().item() < tol_plan:
                break
        return pi, flag

    def find_correspondences(self, normalize=True, norm="l2", bySample=True,
                             k=20, mode="connectivity", metric="correlation",
                             eps=0.01, rho=1.0, rho2=None,
                             nits_plan=500, nits_sinkhorn=500):
        if normalize:
            self.normalize_data(norm=norm, bySample=bySample)
        self.init_marginals()
        print("Computing intra-domain graph distances")
        self.construct_graph(k=k, mode=mode, metric=metric)
        self.init_graph_distances()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GW solver running on: {device}")
        for i in range(len(self.data) - 1):
            print(f"Running pairwise alignment {i + 1}/{len(self.data) - 1}")
            a = torch.Tensor(self.marginals[0]).to(device)
            b = torch.Tensor(self.marginals[i + 1]).to(device)
            dx = torch.Tensor(self.graphDists[0]).to(device)
            dy = torch.Tensor(self.graphDists[i + 1]).to(device)
            coupling, flag = self.exp_unbalanced_gw(a, dx, b, dy, eps=eps, rho=rho, rho2=rho2,
                                                    nits_plan=nits_plan, nits_sinkhorn=nits_sinkhorn)
            self.couplings.append(coupling.cpu())
            self.flags.append(flag)
            if not flag:
                raise Exception(
                    f"Solver got NaN plan with (eps, rho, rho2) = {(eps, rho, rho2)}. "
                    f"Try increasing eps."
                )
        return self.couplings

    def barycentric_projection(self):
        aligned = [self.data[0]]
        for i in range(len(self.couplings)):
            coupling = np.transpose(self.couplings[i].numpy())
            aligned.append(apply_barycentric(coupling, self.data[0]))
        return aligned

    def coembed_datasets(self, Lambda=1.0, out_dim=10):
        """Co-embeds datasets in a shared space (Cao et al. 2022 / Pamona)."""
        n_datasets = len(self.data)
        L = []
        for i in range(n_datasets - 1):
            self.couplings[i] = self.couplings[i] * np.shape(self.data[i])[0]

        for i in range(n_datasets):
            graph_data = (
                self.graphs[i]
                + self.graphs[i].T.multiply(self.graphs[i] < self.graphs[i].T)
                - self.graphs[i].multiply(self.graphs[i] < self.graphs[i].T)
            )
            W = np.array(graph_data.todense())
            W[W > 0] = 1.0 / W[W > 0]
            D = np.diag(W @ np.ones(W.shape[1]))
            L.append(D - W)

        Sigma_x, Sigma_y = [], []
        for i in range(n_datasets - 1):
            Sigma_y.append(np.diag(np.ones(self.couplings[i].shape[0]) @ self.couplings[i]))
            Sigma_x.append(np.diag(self.couplings[i] @ np.ones(self.couplings[i].shape[1])))

        S_xy = self.couplings[0]
        S_xx = L[0] + Lambda * Sigma_x[0]
        S_yy = L[-1] + Lambda * Sigma_y[0]
        for i in range(1, n_datasets - 1):
            S_xy = np.vstack((S_xy, self.couplings[i]))
            S_xx = block_diag(S_xx, L[i] + Lambda * Sigma_x[i])
            S_yy = S_yy + Lambda * Sigma_y[i]

        H_x = sym_inv_sqrt(S_xx)
        H_y = sym_inv_sqrt(S_yy)
        H = H_x @ S_xy @ H_y
        U, _, Vt = np.linalg.svd(H)
        V = Vt.T

        num = [0]
        for i in range(n_datasets - 1):
            num.append(num[i] + len(self.data[i]))

        fx = H_x @ U[:, :out_dim]
        fy = H_y @ V[:, :out_dim]

        integrated = [fx[num[i]:num[i + 1]] for i in range(n_datasets - 1)]
        integrated.append(fy)
        return integrated

    def align(self, normalize=True, norm="l2", bySample=True, k=20,
              mode="connectivity", metric="correlation", eps=0.01, rho=1.0,
              rho2=None, projMethod="embedding", Lambda=1.0, out_dim=10,
              nits_plan=500, nits_sinkhorn=500):
        assert projMethod in ["embedding", "barycentric"]
        self.find_correspondences(normalize=normalize, norm=norm, bySample=bySample,
                                  k=k, mode=mode, metric=metric, eps=eps, rho=rho, rho2=rho2,
                                  nits_plan=nits_plan, nits_sinkhorn=nits_sinkhorn)
        print("FLAGS", self.flags)
        if projMethod == "embedding":
            integrated_data = self.coembed_datasets(Lambda=Lambda, out_dim=out_dim)
        else:
            integrated_data = self.barycentric_projection()
        self.integrated_data = integrated_data
        return integrated_data
