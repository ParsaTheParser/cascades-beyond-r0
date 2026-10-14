import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

# Models for the seed-level take-off curve. Every model maps a vertex to J
# probabilities p_v(tau_1) <= ... <= p_v(tau_J). We get that order for free
# with the same output head everywhere: the model gives J + 1 scores, a
# softmax turns them into the probabilities that the take-off threshold T_v
# falls before tau_1, between tau_1 and tau_2, ..., or after tau_J, and the
# running sum of these is the curve. A take-off probability can then never
# go down when tau goes up, as it must not.


class CurveHead(nn.Module):
    def __init__(self, hidden, J):
        super().__init__()
        self.out = nn.Linear(hidden, J + 1)

    def forward(self, h):
        probs = torch.softmax(self.out(h), dim=-1)
        return torch.cumsum(probs, dim=-1)[..., :-1]


class MLP(nn.Module):
    # Sees only the features of the vertex itself (and the 
    # graph-level inputs), never its neighbours. Used for 
    # the seed-only and the hand-crafted-feature baselines.
    def __init__(self, n_in, J, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.head = CurveHead(hidden, J)

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.head(self.body(x))


class GNN(nn.Module):
    # GIN layers with an edge input (GINE), here whether the two endpoints
    # share a label, so the model knows which edges carry the high bias b^+.
    # After k layers a vertex has seen exactly its k-hop neighbourhood, which
    # is what the depth ladder varies. Each layer adds to the previous state
    # (a residual connection) and is normalised, which keeps deep stacks
    # trainable. Plain message passing of this kind cannot count triangles,
    # so the triangle-aware variant simply gets triangle counts as inputs.
    def __init__(self, n_in, J, depth, hidden=64):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(depth):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=1))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = CurveHead(hidden, J)

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + torch.relu(conv(h, edge_index, edge_attr)))
        return self.head(h)