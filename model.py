import torch
from torch_geometric.nn import CGConv, TransformerConv
from torch.nn import BatchNorm1d, Linear, Softplus, Dropout, ReLU

class CGCNNetL1Sum(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 drop_p=0.3):
        super(CGCNNetL1Sum, self).__init__()
        self.conv = CGConv(channels=num_node_features, 
                           dim=num_edge_features)

        self._activation = ReLU()
        self._dropout=Dropout(drop_p)
        self.dense = Linear(num_node_features, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv(x, edge_index, edge_attr)
        x = self._activation(x)
        x = self._dropout(x)

        x = torch.sum(x, dim=0)

        return self.dense(x)
    
    
class TrfNetL1Sum(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                drop_p=0.3):
        super(TrfNetL1Sum, self).__init__()
        self.conv = TransformerConv(in_channels=num_node_features, out_channels=128,
                                    edge_dim=num_edge_features, 
                                    heads=8, concat=False, aggr="add")

        self._activation = ReLU()
        self._dropout=Dropout(drop_p)
        self.dense = Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv(x, edge_index, edge_attr)
        x = self._activation(x)
        x = self._dropout(x)

        x = torch.sum(x, dim=0)

        return self.dense(x)