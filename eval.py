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
from haversine import haversine
import networkx as nx
import multiprocessing as mp
from scipy import stats
import statistics
import torch_geometric
from termcolor import cprint, colored
import sys
from collections import OrderedDict
import datetime

from model_all import Model
from args import *
from my_constants import *
from utils import *

args = make_args()
print(args)

JUMP = 10000
MAX_ITERS = 300

def trip_length(path):
	global graph, backward
	return sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path])

def intersections_and_unions(path1, path2):
	global graph, backward
	path1, path2 = set(path1), set(path2)
	intersection = sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path1.intersection(path2)])
	union = sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path1.intersection(path2)])
	return intersection, union

def shorten_path(path, true_dest):
	global map_edge_id_to_u_v, backward, map_node_osm_to_coords
	dest_node = map_edge_id_to_u_v[backward[true_dest]][0]
	_, index = min([(haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[edge]][1]], map_node_osm_to_coords[dest_node]), i) for i,edge in enumerate(path)])
	return path[:index+1]

def gen_paths_no_hierarchy(all_paths):
	global JUMP
	ans = []
	for i in tqdm(list(range(0, len(all_paths), JUMP)), desc = "batch_eval", dynamic_ncols=True):
		temp = all_paths[i:i+JUMP]
		ans.append(gen_paths_no_hierarchy_helper(temp))
	return [t for sublist in ans for t in sublist]

def gen_paths_no_hierarchy_helper(all_paths):
	global model, node_nbrs, max_nbrs, edge_to_node_mapping
	true_paths = [p for _,p,_ in all_paths]
	model.eval()
	gens = [[t[0]] for t in true_paths]
	pending = OrderedDict({i:None for i in range(len(all_paths))})
	with torch.no_grad():
		for _ in tqdm(range(MAX_ITERS), desc = "generating trips in lockstep", dynamic_ncols=True):
			true_paths = [all_paths[i][1] for i in pending]
			current_temp = [gens[i][-1] for i in pending]
			assert not args.traffic, colored('Not implemented','red')
			current = [c for c in current_temp for _ in node_nbrs[c]]
			pot_next = [nbr for c in current_temp for nbr in node_nbrs[c]]
			dests = [t[-1] for c,t in zip(current_temp, true_paths) for _ in (node_nbrs[c] if c in node_nbrs else [])]
			unnormalized_confidence = model(current, dests, pot_next)
			chosen = torch.argmax(unnormalized_confidence.reshape(-1, max_nbrs), dim = 1)
			chosen = chosen.detach().cpu().tolist()
			pending_trip_ids = list(pending.keys())
			for identity, choice in zip(pending_trip_ids, chosen):
				choice = node_nbrs[gens[identity][-1]][choice]
				last = gens[identity][-1]
				if choice == -1:
					del pending[identity]
					continue		
				gens[identity].append(choice)
				if choice == all_paths[identity][1][-1]:
					del pending[identity]
			if len(pending) == 0:
				break
			torch.cuda.empty_cache()
	gens = [shorten_path(gen, true[1][-1]) if gen[-1]!=true[1][-1] else gen for gen, true in (zip(gens, all_paths))]
	model.train()
	return gens

def evaluate_no_hierarchy(data, num = 1000, with_dijkstra = False):
	global map_node_osm_to_coords, map_edge_id_to_u_v, backward 
	to_do = ["precision", "recall", "reachability", "avg_reachability", "acc", "nll", "generated"]
	results = {s:None for s in to_do}
	cprint("Evaluating {} number of trips".format(num), "magenta")
	partial = random.sample(data, num)
	t1 = time.time()
	if with_dijkstra:
		gens = [dijkstra(t) for t in tqdm(partial, desc = "Dijkstra for generation", unit = "trip", dynamic_ncols=True)]
	else:
		gens = gen_paths_no_hierarchy(partial)
	elapsed = time.time() -t1
	results["time"] = elapsed
	preserved_with_stamps = partial.copy()
	partial = [p for _,p,_ in partial]
	print("Without correction (everything is weighed according to the edge lengths)")
	generated = list(zip(partial, gens))
	generated = [(t,g) for t,g in generated if len(t)>1]
	lengths = [(trip_length(t), trip_length(g)) for (t,g) in generated]
	inter_union = [intersections_and_unions(t, g) for (t,g) in generated]
	inters = [inter for inter,union in inter_union]
	lengths_gen = [l_g for l_t,l_g in lengths]
	lengths_true = [l_t for l_t,l_g in lengths]
	precs = [i/l if l >0 else 0 for i,l in zip(inters, lengths_gen) ]
	precision1 = round(100*sum(precs)/len(precs), 2)
	recs = [i/l if l >0 else 0 for i,l in zip(inters, lengths_true) ]
	recall1 = round(100*sum(recs)/len(recs), 2)
	deepst_accs = [i/max(l1,l2) for i,l1,l2 in zip(inters, lengths_true, lengths_gen) if max(l1,l2)>0]
	deepst = round(100*sum(deepst_accs)/len(deepst_accs), 2)
	num_reached = len([None for t,g in generated if t[-1] == g[-1]])
	lefts = [haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[g[-1]]][0]], map_node_osm_to_coords[map_edge_id_to_u_v[backward[t[-1]]][0]]) for t,g in generated]
	rights = [haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[g[-1]]][1]], map_node_osm_to_coords[map_edge_id_to_u_v[backward[t[-1]]][1]]) for t,g in generated]
	reachability = [(l+r)/2 for (l,r) in zip(lefts,rights)]
	all_reach = np.mean(reachability)
	all_reach = round(1000*all_reach,2)
	
	if len(reachability) != num_reached:
		reach = sum(reachability)/(len(reachability)-num_reached)
	else:
		reach = 0
	reach = round(1000*reach,2)
	percent_reached = round(100*(num_reached/len(reachability)), 2)
	print()
	cprint("Precision is                            {}%".format(precision1), "green")
	cprint("Recall is                               {}%".format(recall1), "green")
	print()
	cprint("%age of trips reached is                {}%".format(percent_reached), "green")
	cprint("Avg Reachability(across all trips) is   {}m".format(all_reach), "green")
	print()
	results["precision"] = precision1
	results["reachability"] = percent_reached
	results["avg_reachability"] = (all_reach, reach)
	results["recall"] = recall1
	results["generated"] = list(zip(preserved_with_stamps, gens))
	return results

def load_model(path_model, path_extras):
	f = open(path_extras, 'rb')
	forward, map_node_osm_to_coords, map_edge_id_to_u_v = pickle.load(f)
	f.close()
	backward = {forward[k]:k for k in forward}
	node_nbrs = create_node_nbrs(forward)
	transformed_graph = nx.DiGraph()
	for e1 in node_nbrs:
		for e2 in node_nbrs[e1]:
			if e2 != -1:
				transformed_graph.add_edge(e1, e2)
	print('Support file loaded')
	model = torch.load(path_model)
	model.eval()
	print('Model loaded')
	return model, map_node_osm_to_coords, map_edge_id_to_u_v, forward, backward, node_nbrs, transformed_graph

def load_data_and_test(path_data):
	global model, map_node_osm_to_coords, map_edge_id_to_u_v, forward, backward, node_nbrs, transformed_graph
	data = load_test_data(args, forward, fname = path_data)
	results = evaluate_no_hierarchy(data = data, num =len(data), with_dijkstra = False)
	return results

def dijkstra(true_trip):
	global args, transformed_graph, max_nbrs, model
	assert args.loss == "v2", "I dont think this will work for loss v1"
	_, (src, *_, dest), (s, _) = true_trip
	g = transformed_graph
	with torch.no_grad():
		current_temp = [c for c in g.nodes()]
		current = [c for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else []) ]
		pot_next = [nbr for c in current_temp for nbr in (node_nbrs[c] if c in node_nbrs else [])]
		dests = [dest for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else [])]
		traffic = None
		unnormalized_confidence = model(current, dests, pot_next, traffic)
		unnormalized_confidence = -1*torch.nn.functional.log_softmax(unnormalized_confidence.reshape(-1, max_nbrs), dim = 1)
		transition_nll = unnormalized_confidence.detach().cpu().tolist()
	torch.cuda.empty_cache()
	count = 0
	for u in g.nodes():
		for i,nbr in enumerate(node_nbrs[u]):
			if nbr == -1:
				break
			g[u][nbr]["nll"] = transition_nll[count][i]
		count += 1
	path =  nx.dijkstra_path(g, src, dest, weight = "nll")
	path = [x for x in path]
	return path

if __name__ == "__main__":
	model, map_node_osm_to_coords, map_edge_id_to_u_v, forward, backward, node_nbrs, transformed_graph = load_model(path_model = MODEL_SAVE_PATH, path_extras = MODEL_SUPPORT_PATH)
	run_dijkstra = args.with_dijkstra
	if run_dijkstra:
		cprint("Running NeuroMLR-Dijkstra", "yellow")
	else:
		cprint("Running NeuroMLR-Greedy", "yellow")
	args = model.args
	print(model.device)
	model.eval()

	f = open(PICKLED_GRAPH,'rb')
	graph = pickle.load(f)
	for e in graph.edges(data=True):
		e[2]['length'] = e[2]['length']/1000
	f.close()
	max_nbrs = max(len(nbr_array) for nbr_array in node_nbrs.values())
	for u in range(len(forward)):
		if u in node_nbrs:
			node_nbrs[u].extend([-1]*(max_nbrs - len(node_nbrs[u])))
		else:
			node_nbrs[u] = [-1]*max_nbrs
	nodes_used = set()
	for e in forward:
		u,v = map_edge_id_to_u_v[e]
		nodes_used.add(u)
		nodes_used.add(v)
	nodes_used = list(nodes_used)
	nodes_forward = {node:i for i,node in enumerate(nodes_used)}
	edge_to_node_mapping = {forward[e]:(nodes_forward[map_edge_id_to_u_v[e][0]], nodes_forward[map_edge_id_to_u_v[e][1]]) for e in forward}
	edge_to_node_mapping[-1] = (-1,-1)

	transformed_graph = nx.DiGraph()  # required for Dijkstra
	for e1 in node_nbrs:
		for e2 in node_nbrs[e1]:
			if e2 != -1:
				transformed_graph.add_edge(e1, e2)
	
	# data = load_test_data(args, forward, fname = TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
	data = load_test_data(args, forward, fname = TEST_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS)
	if run_dijkstra:
		cprint("Running Dijkstra takes long. Consider sampling a small test set, if required", "red")
		data = random.sample(data, 500)
	results = evaluate_no_hierarchy(data = data, num = min(10000,len(data)), with_dijkstra = run_dijkstra)