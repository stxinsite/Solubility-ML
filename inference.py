DEV_DIR = "/bgfs01/insite/olga.kononova/dev/"
import sys
sys.path.append(DEV_DIR)

import json
import os
import pandas as pd
import plotly.graph_objects as go

from progressbar import ProgressBar

import torch
from torch_geometric.data import Data

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MolFromSmiles, AllChem

from sklearn.metrics import mean_absolute_error, mean_squared_error

from model import TrfNetL1Sum

from GNN_models.gnn_parent import GNNEdgeWrapper
from GNN_models.trf_models import TrfEdgeNet
from GNN_models.features import get_atomic_features, get_edges_from_bonds, get_features_dim

dirname = os.path.dirname(__file__)
print(dirname)

num_node_features, num_edge_features = get_features_dim()
num_classes = 492
embed_model = GNNEdgeWrapper(TrfEdgeNet(num_node_features, 
                                        num_edge_features, 
                                        num_classes),
                             model_path=os.path.join(dirname, "trained_models/"),
                             name="TrfEdge",
                             suffix="wts_04132022_run",
                             device="cpu"
                            )
embed_model.load_model()
embed_model.model.eval()

input_size = 256
sol_model = TrfNetL1Sum(input_size, num_edge_features)
sol_model.load_state_dict(torch.load(os.path.join(dirname, "trained_models/model_Trf_MSEloss_1.pt")))
sol_model.eval()

dataset = pd.read_csv(os.path.join(dirname, "data/soly_Test.csv"))
print(len(dataset))

bar = ProgressBar(max_value=len(dataset))
output = []
for num, (k_smile, c_smile, logS) in enumerate(zip(dataset["Kekule_smiles"], dataset["SMILES_stand"], dataset["logS"])):
    mol = MolFromSmiles(k_smile)
    if not mol:
        mol = MolFromSmiles(c_smile)
        if not mol:
            print(k_smile, c_smile)
            continue
    mol = AllChem.AddHs(mol)
    x = [get_atomic_features(a) for a in mol.GetAtoms()]
    start_idx, end_idx, edge = get_edges_from_bonds(mol.GetBonds())
    torch_vector = embed_model.features_to_torch_vec(dict(x=[get_atomic_features(a) for a in mol.GetAtoms()],
                                                              y=[],
                                                              edge_attr=edge,
                                                              edge_index=[start_idx, end_idx]))
    embeddings_vec = embed_model.get_embeddings(torch_vector).tolist()
    torch_data = Data(x=torch.tensor(embeddings_vec, dtype=torch.float),
                      edge_attr=torch.tensor(edge, dtype=torch.float),
                      edge_index=torch.tensor([start_idx, end_idx], dtype=torch.long)
                     )
    
    prediction = sol_model(torch_data).item()
    output.append((prediction, logS))
        
    bar.update(num)
    
print("\n")
print("MAE:", mean_absolute_error([p for p, y in output],[y for p, y in output]))
print("MSE:", mean_squared_error([p for p, y in output],[y for p, y in output]))

