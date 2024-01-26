"""Euclidean Knowledge Graph embedding models where embeddings are in complex space."""
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance
import torch.nn.functional as F
QUATERNION_MODELS = ["SimiKE"]


class BaseQ(KGModel):
    """Quaternion Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args):
        """Initialize a Quaternion KGModel."""
        super(BaseQ, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score

    def get_factors(self, queries):
        """Compute factors for embeddings' regularization."""
        lhs = self.entity(queries[:, 0])
        rel = self.rel(queries[:, 1])
        rel_diag = self.rel_diag(queries[:, 1])
        rhs = self.entity(queries[:, 2])

        head_e = torch.chunk(lhs, 4, dim=1)
        rel_e = torch.chunk(rel, 4, dim=1)
        rhs_e = torch.chunk(rhs, 4, dim=1)
        rel_diag_e = torch.chunk(rhs, 4, dim=1)

        head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2 + head_e[2] ** 2 + head_e[3] ** 2)
        rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2 + rel_e[2] ** 2 + rel_e[3] ** 2)
        rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2 + rhs_e[2] ** 2 + rhs_e[3] ** 2)
        rel_diag_f = torch.sqrt(rel_diag_e[0] ** 2 + rel_diag_e[1] ** 2 + rel_diag_e[2] ** 2 + rel_diag_e[3] ** 2)
        return head_f, rel_f, rhs_f, rel_diag_f

class SimiKE(BaseQ):
    def __init__(self, args):
        super(SimiKE, self).__init__(args)
        self.sim = "dist"

        self.entity = nn.Embedding(self.sizes[0], 4 * self.rank)
        self.rel = nn.Embedding(self.sizes[1], 4 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], 4 * self.rank)

        self.entity.weight.data = self.init_size * self.entity.weight.to(self.data_type)
        self.rel.weight.data = self.init_size * self.rel.weight.to(self.data_type)
        self.rel_diag.weight.data = self.init_size * self.rel_diag.weight.to(self.data_type)

    def q_norm(self, r):
        s_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=1)
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b
        return torch.cat([s_b, x_b, y_b, z_b], dim=1)

    def quaternion_mul(self, rel, lhs):
        a, b, c, d = torch.chunk(rel, 4, dim=1)
        e, f, g, h = torch.chunk(lhs, 4, dim=1)
        return torch.cat(
            [(a * e - b * f - c * g - d * h),
             (b * e + a * f + c * h - d * g),
             (c * e + a * g + d * f - b * h),
             (d * e + a * h + b * g - c * f)], dim=1)

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs = self.entity(queries[:, 0])
        rel = self.rel(queries[:, 1])
        rhs = self.entity(queries[:, 2])
        rel_diag = self.rel_diag(queries[:, 1])
        # print(lhs.shape, rel_diag.shape)

        rel_diag = self.q_norm(rel_diag)
        lhs_e = self.quaternion_mul(rel_diag, lhs)

        a, b, c, d = torch.chunk(lhs_e, 4, dim=1)
        e, f, g, h = torch.chunk(rel, 4, dim=1)
        i, j, k, l = torch.chunk(rhs, 4, dim=1)

        theta1 = F.cosine_similarity(a, i, dim=-1).unsqueeze(1)
        theta2 = F.cosine_similarity(b, j, dim=-1).unsqueeze(1)
        theta3 = F.cosine_similarity(c, k, dim=-1).unsqueeze(1)
        theta4 = F.cosine_similarity(d, l, dim=-1).unsqueeze(1)
        # theta1 = F.cosine_similarity(a, i, dim=-1).unsqueeze(1)
        # theta2 = F.cosine_similarity(b, j, dim=-1).unsqueeze(1)
        # theta3 = F.cosine_similarity(c, k, dim=-1).unsqueeze(1)

        e = e * (1 - theta1)
        f = f * (1 - theta2)
        g = g * (1 - theta3)
        h = h * (1 - theta4)

        new_rel = torch.cat([e, f, g, h], dim=1)

        lhs_e = lhs_e + new_rel
        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases
