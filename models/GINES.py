from typing import Callable, Optional, Union

import torch
from torch import Tensor


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import reset

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_add_pool



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


from utils.utils import pair_norm

class GINEConv_modify(MessagePassing):

    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'std')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()
        self.lin2 = Linear(in_channels, in_channels)

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (self.lin2(x_j - x_i) + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'








class GINES(torch.nn.Module):
    def __init__(self, hidden_channels, aggr_method):
        super(GINES, self).__init__()
        #         torch.manual_seed(12345)
        self.conv1 = GINEConv_modify(
            Sequential(Linear(2, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()), edge_dim=1, aggr=aggr_method)
        # Applies batch normalization over a batch of node features
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/norm/batch_norm.html
        #         self.bn1 = BatchNorm(hidden_channels)
        self.bn1 = pair_norm()
        #         self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #         self.bn2 = BatchNorm(hidden_channels)

        self.conv2 = GINEConv_modify(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()), edge_dim=1, aggr=aggr_method)
        #         self.bn3 = BatchNorm(hidden_channels)
        self.bn2 = pair_norm()

        self.conv3 = GINEConv_modify(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()), edge_dim=1, aggr=aggr_method)
        #         self.bn3 = BatchNorm(hidden_channels)
        self.bn3 = pair_norm()

        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 3)
        self.lin2 = Linear(hidden_channels * 3, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        h1 = self.conv1(x, edge_index, edge_attr).relu()
        h1 = self.bn1(h1)

        h2 = self.conv2(h1, edge_index, edge_attr).relu()
        h2 = self.bn2(h2)

        h3 = self.conv3(h2, edge_index, edge_attr).relu()
        h3 = self.bn3(h3)

        h1 = global_add_pool(h1, batch)  # [batch_size, hidden_channels]
        h2 = global_add_pool(h2, batch)  # [batch_size, hidden_channels]
        h3 = global_add_pool(h3, batch)  # [batch_size, hidden_channels]

        h = torch.cat((h1, h2, h3), dim=1)
        # 3. Apply a final classifier
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=-1)


# hidden_channels = 32
# aggr = 'std'
# model = GINES(hidden_channels=hidden_channels, aggr_method=aggr)
#
# print(model)