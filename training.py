DEV_DIR = "/bgfs01/insite/olga.kononova/dev/"
WORK_DIR = "/bgfs01/insite/olga.kononova/solubility"
import sys
sys.path.append(DEV_DIR)

import json
import os
import pandas as pd
import numpy as np
from progressbar import ProgressBar

import torch
#import torch.utils.data as Data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader
DEVICE = torch.device("cuda")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MolFromSmiles, AllChem

from model import CGCNNetL1Sum, TrfNetL1Sum
from utils import train, test

from GNN_models.gnn_parent import GNNEdgeWrapper
from GNN_models.trf_models import TrfEdgeNet
from GNN_models.features import get_atomic_features, get_edges_from_bonds, get_features_dim

atom_types_dict = json.loads(open(os.path.join(DEV_DIR, "ResourceFiles/atom_labels_dict_09142021.json")).read())
num_to_atom_types = {i: c for c, i in atom_types_dict.items()}

num_node_features, num_edge_features = get_features_dim()
num_classes = len(atom_types_dict)

print("Number of node features:", num_node_features)
print("Number of edge features:", num_edge_features)
print("Number of classes:", num_classes)

models_dir = "/bgfs01/insite/olga.kononova/atom_types_prediction/"
embedding_model = GNNEdgeWrapper(TrfEdgeNet(num_node_features, 
                                            num_edge_features, 
                                            num_classes),
                                 model_path=os.path.join(models_dir, "TrfEdge"),
                                 name="TrfEdge",
                                 suffix="wts_04132022_run",
                                 device="cpu"
                                 )
embedding_model.load_model()
embedding_model.model.eval()
print(embedding_model.model)

dataset = pd.read_csv(os.path.join(WORK_DIR, "sol_clean_v1_train.csv"))
print(len(dataset))

print("Creating training set...")
training_set = []
x, y = [], []
bar = ProgressBar(max_value=len(dataset))
for num, (k_smile, c_smile, logS) in enumerate(zip(dataset["kekSmiles"], dataset["SMILES_stand"], dataset["logS"])):
    mol = MolFromSmiles(k_smile)
    if not mol:
        mol = MolFromSmiles(c_smile)
        if not mol:
            print(k_smile, c_smile)
            continue
    mol = AllChem.AddHs(mol)
    start_idx, end_idx, edge = get_edges_from_bonds(mol.GetBonds())
    torch_vector = embedding_model.features_to_torch_vec(dict(x=[get_atomic_features(a) for a in mol.GetAtoms()],
                                                              y=[],
                                                              edge_attr=edge,
                                                              edge_index=[start_idx, end_idx]))
    embeddings_vec = embedding_model.get_embeddings(torch_vector)#.tolist()
    training_set.append(Data(x=embeddings_vec,
                             #x=torch.tensor(embeddings_vec, dtype=torch.float),
                             #x=torch.tensor([get_atomic_features(a) for a in mol.GetAtoms()], dtype=torch.float),
                             y=torch.tensor([logS], dtype=torch.float),
                             edge_attr=torch.tensor(edge, dtype=torch.float),
                             edge_index=torch.tensor([start_idx, end_idx], dtype=torch.long)
                             ))
    
    bar.update(num)
    #if num > 10:
    #    break
print("\n")

input_size = training_set[0].x.size(1)
print("Model input size:", input_size)
#sol_model = CGCNNetL1Sum(input_size, num_edge_features).to(DEVICE)
sol_model = TrfNetL1Sum(input_size, num_edge_features).to(DEVICE)
sol_model.train()
print(sol_model)

optimizer = torch.optim.Adam(sol_model.parameters(), lr=0.001)
#loss_func = torch.nn.L1Loss()
loss_func = torch.nn.MSELoss()
batch_size = 128

# training_loader = DataLoader(training_set,
#                              batch_size=1,
#                              shuffle=False)

#for batch in training_loader:
#   print(batch)
#    print(model(d.to(DEVICE)).size(), d.y.size())
    
epochs_num = 200
loss_output = []
bar = ProgressBar(max_value=epochs_num)
for epoch in range(epochs_num):
    subset = [training_set[i] 
              for i in np.random.choice(len(training_set)-1, batch_size)]
    training_loader = DataLoader(subset,
                                 batch_size=1,
                                 shuffle=False)
    loss = train(sol_model, training_loader, loss_func, optimizer, DEVICE)
    loss_output.append(loss)
    bar.update(epoch)
    #print(f"Epoch: {epoch}, loss {loss}")
    
training_loader = DataLoader(training_set,
                             batch_size=1,
                             shuffle=False)
loss = test(sol_model, training_loader, loss_func)
print("Test loss:", loss)

# with open("output/Trf_MSEloss_discrt.json", "w") as f:
#     f.write(json.dumps(loss_output))
# torch.save(sol_model.state_dict(), "output/model_Trf_MSEloss_discrt.pt")
    
print("Done!")