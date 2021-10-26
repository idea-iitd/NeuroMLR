import numpy as np
import pickle
from tqdm import tqdm
import torch
from torch import nn 
import pandas as pd
import geopandas as gpd
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
import time
from my_constants import *
from collections import defaultdict
import networkx as nx

edge_df = gpd.read_file(EDGE_DATA)
node_df = gpd.read_file(NODE_DATA)
a = node_df['osmid'].to_numpy()
b = node_df[['y', 'x']].to_numpy()
map_node_osm_to_coords = {e:(u,v) for e,(u,v) in zip(a,b)}
del a
del b

map_edge_id_to_u_v = edge_df[['u', 'v']].to_numpy()
map_u_v_to_edge_id = {(u,v):i for i,(u,v) in enumerate(map_edge_id_to_u_v)}
unique_edge_labels = list(map_u_v_to_edge_id.values())		# these are from OSM map data, not train data

def condense_edges(edge_route):
	global map_edge_id_to_u_v, map_u_v_to_edge_id
	route = [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in edge_route]
	return route

def fetch_map_fid_to_zero_indexed(data):
	s = set()
	for _,t,_ in data:
		s.update(set(t))
	return {el:i for i,el in enumerate(s)}

def relabel_trips(data, mapping):
	return [(idx, [mapping[e] for e in trip], timestamps) for (idx, trip, timestamps) in data]

def remove_loops(path):
	reduced = []
	last_occ = {p:-1 for p in path}
	for i in range(len(path)-1,-1,-1):
		if last_occ[path[i]] == -1:
			last_occ[path[i]] = i
	current = 0
	while(current < len(path)):
		reduced.append(path[current])
		current = last_occ[path[current]] + 1
	return reduced

def nbrs_sanity_check(node_nbrs, data):
	print("SANITY CHECK 1")
	for _, t, _ in tqdm(data, dynamic_ncols=True):
		for i in range(len(t)-1):
			assert t[i+1] in node_nbrs[t[i]], "How did this happen?"
	print('Cleared :)')

def create_node_nbrs(forward):
	start_nodes = defaultdict(set)
	for e in forward:
		u,v = map_edge_id_to_u_v[e]
		start_nodes[u].add(forward[e])
	node_nbrs = {}	# here nodes are actually edges of the road network
	for e in forward:
		_,v = map_edge_id_to_u_v[e]
		node_nbrs[forward[e]] = list(start_nodes[v])
	return node_nbrs

def load_data(args, less=False, sample=1000, fname=TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS):
	print("Loading map matched trajectories")
	f = open(fname, "rb")
	data = pickle.load(f)
	f.close()
			
	if less:
		data = data[:sample]

	data = [(idx, condense_edges(t), timestamps) for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
	
	if args.remove_loops or args.remove_loops_from_train:
		data = [(idx, remove_loops(t),timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True)]
		
	data = [(idx,t,timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True) if len(t) >= 5]	# ignoring very small trips
	
	forward = fetch_map_fid_to_zero_indexed(data)	

	data = relabel_trips(data, forward)
	return data, forward

def load_test_data(args, forward, less=False, sample = 1000, fname = TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS):
	print('Loading test/val data')
	f = open(fname, "rb")
	data = pickle.load(f)
	f.close()
	if less:
		data = data[:sample]
	data = [(idx, condense_edges(t), timestamps) for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
	if args.remove_loops:
		data = [(idx, remove_loops(t),timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True)]
	data = [(idx,t,timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True) if len(t) >= 5]	# ignoring very small trips

	orig_num = len(data)
	print('Number of trips in test data (initially): {}'.format(orig_num))
	unseen_data = [trip_tup for trip_tup in data if not set(trip_tup[1]).issubset(forward)]
	for (_, trip, _) in unseen_data:
		for e in trip:
			if e not in forward:
				forward[e] = len(forward)

	print('Number of trips with unseen nodes: {}'.format(len(unseen_data)))
	print('Keeping these trips!')
	print('Relabelling trips with updated forward map')
	data = relabel_trips(data, forward)
	return data

def single_source_shortest_path_length_range(graph, node_range, cutoff):
	dists_dict = {}
	for node in node_range:
		dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight='haversine')
	return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result
