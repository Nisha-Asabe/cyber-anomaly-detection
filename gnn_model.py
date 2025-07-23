import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F

class GCNAnomalyDetector(torch.nn.Module):
    def __init__(self, input_dim):
        super(GCNAnomalyDetector, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def build_graph(df):
    from torch_geometric.utils import dense_to_sparse
    import torch
    # Use correlation as adjacency
    A = torch.tensor(df.corr().abs().values, dtype=torch.float)
    edge_index, _ = dense_to_sparse(A)
    x = torch.tensor(df.values, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)
