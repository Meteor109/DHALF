from torch.nn import Sequential, Linear, ReLU
from dgl.nn import EdgeWeightNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from utils import *

EOS = 1e-10
norm = EdgeWeightNorm(norm='both')


class GCL(nn.Module):
    def __init__(self, nlayers, nlayers_proj, in_dim, emb_dim, proj_dim, dropout, sparse,
                 gate_hidden=64,
                 full_mlp_hidden=256, full_mlp_out=None, variant_d_gate_mode="node"):
        super(GCL, self).__init__()

        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)

        if nlayers_proj == 1:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim))
        elif nlayers_proj == 2:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

        self.full_mlp_out = proj_dim if full_mlp_out is None else full_mlp_out
        self.variant_d_gate_mode = variant_d_gate_mode

        # Z = alpha * Z_L + beta * Z_H, with softmax keeping alpha, beta > 0 and summing to 1
        self.coef_logits = nn.Parameter(torch.zeros(2))  # softmax -> [α, β]

        # Discriminator D(z_i, g) = sigmoid(z_i W g^T)
        self.disc_W = nn.Parameter(torch.empty(proj_dim, proj_dim))
        nn.init.xavier_uniform_(self.disc_W)

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

        self.full_mlp = Sequential(
            Linear(in_dim, full_mlp_hidden),
            ReLU(inplace=True),
            Linear(full_mlp_hidden, self.full_mlp_out),
        )
        if self.full_mlp_out == proj_dim:
            self.full_align = nn.Identity()
        else:
            self.full_align = Linear(self.full_mlp_out, proj_dim)

        if self.variant_d_gate_mode == "scalar":
            self.gate_logit = nn.Parameter(torch.zeros(1))
            self.gate_mlp = None
        else:
            self.gate_logit = None
            self.gate_mlp = Sequential(
                Linear(proj_dim * 2, gate_hidden),
                ReLU(inplace=True),
                Linear(gate_hidden, 1),
            )


    def get_embedding(self, x, a1, a2, source='all'):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        return torch.cat((emb1, emb2), dim=1)


    def get_variant_d_embeddings(self, x, a1, a2):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        z_low = self.proj_head1(emb1)
        z_high = self.proj_head2(emb2)
        z_full = self.full_mlp(x)
        if self.variant_d_gate_mode == "scalar":
            gate = torch.sigmoid(self.gate_logit).view(1, 1).expand(z_low.shape[0], 1)
        else:
            gate = torch.sigmoid(self.gate_mlp(torch.cat((z_low, z_high), dim=1)))
        z_mix = gate * z_low + (1.0 - gate) * z_high
        return {
            "z_low": z_low,
            "z_high": z_high,
            "z_full": z_full,
            "z_full_loss": self.full_align(z_full),
            "z_mix": z_mix,
            "gate": gate,
        }

    def _combine(self, zL: torch.Tensor, zH: torch.Tensor):
        """Z = α Z^L + β Z^H; g = mean(Z)."""
        coef = torch.softmax(self.coef_logits, dim=0)  # [α, β]
        Z = coef[0] * zL + coef[1] * zH               # (N,d)
        g = Z.mean(dim=0)                              # (d,)
        return Z, g, coef

    def _disc_logits(self, z: torch.Tensor, g: torch.Tensor):
        """logits_i = z_i W g^T, shape (N,)"""
        wg = self.disc_W @ g           # (d,)
        return z @ wg                  # (N,)

    def original_discriminator_loss(self, zL, zH, zL_neg, zH_neg):
        _, g, _ = self._combine(zL, zH)

        pos_L = self._disc_logits(zL, g)
        neg_L = self._disc_logits(zL_neg, g)
        pos_H = self._disc_logits(zH, g)
        neg_H = self._disc_logits(zH_neg, g)

        ones = torch.ones_like(pos_L)
        zeros = torch.zeros_like(neg_L)

        loss_L = self.bce(torch.cat([pos_L, neg_L], dim=0),
                          torch.cat([ones, zeros], dim=0))
        loss_H = self.bce(torch.cat([pos_H, neg_H], dim=0),
                          torch.cat([ones, zeros], dim=0))

        return (loss_L + loss_H) / 2.0

    def variant_d_contrastive_loss(self, anchor, sample, pos_mask, tau=0.5):
        pos_mask = pos_mask.to(anchor.device).float()
        anchor = F.normalize(anchor, p=2, dim=1)
        sample = F.normalize(sample, p=2, dim=1)
        logits = anchor @ sample.t() / tau
        log_prob = F.log_softmax(logits, dim=1)
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return logits.sum() * 0.0
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_count.clamp_min(1.0)
        return loss[valid].mean()

    def variant_d_mix_loss(self, emb, fh_pos_mask, tau=0.5, detach_branches=True, anchor=None):
        """Fusion-consistency objective that trains the Eq.(12) node gate.

        The gate is part of the fused readout, so it needs an objective that depends on
        Z_mix. This term anchors the fusion to the graph-free semantic neighbourhood
        structure M_FH, which is what makes the node-wise routing meaningful: a node whose
        neighbourhood is semantically coherent can keep its low-pass evidence, while a
        boundary node keeps more of its high-pass evidence.

        `detach_branches=True` keeps L_FL/L_FH as the only supervision of the two
        encoders, so L_MIX updates the gate parameters and nothing else.

        `anchor` overrides the default Z_full anchor. The "no auxiliary branch" control
        uses the detached branch consensus instead, so the gate stays adaptive even when
        the full-pass view is absent.
        """
        z_low = emb["z_low"].detach() if detach_branches else emb["z_low"]
        z_high = emb["z_high"].detach() if detach_branches else emb["z_high"]
        # Re-compute the gate from the (possibly detached) branch embeddings instead of
        # reusing emb["gate"], which was built on the non-detached embeddings and would
        # otherwise leak gradient from L_MIX back into the encoders even when
        # detach_branches is set. The forward value is identical; only the gradient path
        # changes, so that L_MIX updates the gate parameters and nothing else.
        gate = torch.sigmoid(self.gate_mlp(torch.cat((z_low, z_high), dim=1)))
        z_mix = gate * z_low + (1.0 - gate) * z_high
        if anchor is None:
            z_anchor = emb["z_full_loss"].detach() if detach_branches else emb["z_full_loss"]
        else:
            z_anchor = anchor.detach() if detach_branches else anchor
        return 0.5 * (
            self.variant_d_contrastive_loss(z_anchor, z_mix, fh_pos_mask, tau=tau)
            + self.variant_d_contrastive_loss(z_mix, z_anchor, fh_pos_mask.t(), tau=tau)
        )

    def variant_d_branch_consensus(self, emb):
        """Detached mean of the two frequency-branch views, used as a fallback anchor."""
        return 0.5 * (emb["z_low"].detach() + emb["z_high"].detach())

    def variant_d_losses(self, x, a1, a2, fl_pos_mask, fh_pos_mask, tau=0.5):
        emb = self.get_variant_d_embeddings(x, a1, a2)
        z_full_loss = emb["z_full_loss"]
        l_fl = 0.5 * (
            self.variant_d_contrastive_loss(z_full_loss, emb["z_low"], fl_pos_mask, tau=tau)
            + self.variant_d_contrastive_loss(emb["z_low"], z_full_loss, fl_pos_mask.t(), tau=tau)
        )
        l_fh = 0.5 * (
            self.variant_d_contrastive_loss(z_full_loss, emb["z_high"], fh_pos_mask, tau=tau)
            + self.variant_d_contrastive_loss(emb["z_high"], z_full_loss, fh_pos_mask.t(), tau=tau)
        )
        return l_fl, l_fh, emb


    def forward(self, x1, a1, x2, a2):
        emb1 = self.encoder1(x1, a1)
        emb2 = self.encoder2(x2, a2)
        proj1 = self.proj_head1(emb1)   # Z^L
        proj2 = self.proj_head2(emb2)   # Z^H

        perm = torch.randperm(x1.size(0), device=x1.device)
        x1_corrupt = x1[perm]
        x2_corrupt = x2[perm]  # shuffle rows: replace "feature of spot i" with "feature of another random spot"

        emb1_neg = self.encoder1(x1_corrupt, a1)
        emb2_neg = self.encoder2(x2_corrupt, a2)
        proj1_neg = self.proj_head1(emb1_neg)
        proj2_neg = self.proj_head2(emb2_neg)

        return self.original_discriminator_loss(proj1, proj2, proj1_neg, proj2_neg)


class Edge_Discriminator(nn.Module):
    def __init__(self, nnodes, input_dim, alpha, sparse, hidden_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)

        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.sparse = sparse
        self.alpha = alpha


    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h


    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2


    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()


    # def weight_forward(self, features, edges):
    #     embeddings = self.get_node_embedding(features)
    #     edges_weights_raw = self.get_edge_weight(embeddings, edges)
    #     weights_lp = self.gumbel_sampling(edges_weights_raw)
    #     weights_hp = 1 - weights_lp
    #     return weights_lp, weights_hp

    def weight_forward(self, features, edges, force_uniform=False):
        if force_uniform:
            num_edges = edges.shape[1]
            weights_lp = torch.ones(num_edges, device=features.device)
            weights_hp = torch.zeros(num_edges, device=features.device)
            return weights_lp, weights_hp

        embeddings = self.get_node_embedding(features)
        edges_weights_raw = self.get_edge_weight(embeddings, edges)
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp



    def weight_to_adj(self, edges, weights_lp, weights_hp):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, weights_lp, self.nnodes)
            adj_lp += torch.eye(self.nnodes).cuda()
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)

            adj_hp = get_adj_from_edges(edges, weights_hp, self.nnodes)
            adj_hp += torch.eye(self.nnodes).cuda()
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)

            mask = torch.zeros(adj_lp.shape).cuda()
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes).cuda() - adj_hp * mask * self.alpha
        else:
            adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_lp = dgl.add_self_loop(adj_lp)
            weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes).cuda())) + EOS
            weights_lp = norm(adj_lp, weights_lp)
            adj_lp.edata['w'] = weights_lp

            adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_hp = dgl.add_self_loop(adj_hp)
            weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes).cuda())) + EOS
            weights_hp = norm(adj_hp, weights_hp)
            weights_hp *= - self.alpha
            weights_hp[edges.shape[1]:] = 1
            adj_hp.edata['w'] = weights_hp
        return adj_lp, adj_hp


    # def forward(self, features, edges):
    #     weights_lp, weights_hp = self.weight_forward(features, edges)
    #     adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)
    #     return adj_lp, adj_hp, weights_lp, weights_hp

    def forward(self, features, edges, force_uniform=False):
        weights_lp, weights_hp = self.weight_forward(features, edges, force_uniform)
        adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)
        return adj_lp, adj_hp, weights_lp, weights_hp



class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse

        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

    def forward(self, x, g):
        x = torch.relu(self.linear(x))

        if self.sparse:
            with g.local_scope():
                g.ndata['h'] = x
                for _ in range(self.k):
                    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                return g.ndata['h']
        else:
            for _ in range(self.k):
                x = torch.matmul(g, x)
            return x
