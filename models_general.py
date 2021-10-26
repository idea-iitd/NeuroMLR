import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GNN(torch.nn.Module):
	def __init__(self, node_feature_size, output_embedding_size, num_layers, hidden_dim, graph, gnn_type = "GCN"):
		super(GNN, self).__init__()
		layer_sizes = [node_feature_size] + [hidden_dim]*(num_layers) + [output_embedding_size]

		if gnn_type == "GCN":
			print("Using a GCN")
			layer_function = GCNConv
		elif gnn_type == "GAT":
			print("Using a GAT network")
			layer_function = GATConv
		elif gnn_type == "PGNN":
			print("Using PGNN")
			print("NOT IMPLEMENTED")
			raise SystemExit

		self.layers = nn.ModuleList([layer_function(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])
		self.data = graph

	def forward(self):
		x, edge_index = self.data.x, self.data.edge_index

		for i,neighbour_agg in enumerate(self.layers):
			x = neighbour_agg(x, edge_index)
			if i!= len(self.layers) - 1:
				x = F.relu(x)
		return x



class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, num_layers, hidden_dim):

		assert num_layers >= 0 , "invalid input"
		super(MLP, self).__init__()
		layer_sizes = [input_dim] + [hidden_dim]*(num_layers) + [output_dim]
		self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])
		self.non_linearity = nn.ReLU()


	def forward(self, x):
		for i,linear_tranform in enumerate(self.layers):
			x = linear_tranform(x)
			if i!= len(self.layers) - 1:
				x = F.relu(x)
		return x