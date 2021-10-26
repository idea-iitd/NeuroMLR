from termcolor import cprint
import pickle
from my_constants import *
import geopandas as gpd
import numpy as np
from haversine import haversine
from tqdm import tqdm
from datetime import datetime
from sklearn.decomposition import PCA
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from termcolor import colored, cprint

def find_interval_1(end_time):
	return str(datetime.utcfromtimestamp(end_time)).split()[0], datetime.utcfromtimestamp(end_time).hour 

def find_interval_2(end_time):
	return (-1,datetime.utcfromtimestamp(end_time).hour)

if args.check_script:
	DATA_THRESHOLD = 10
else:
	DATA_THRESHOLD = 200


edge_df = gpd.read_file(EDGE_DATA)
node_df = gpd.read_file(NODE_DATA) 
map_node_osm_to_coords = { node_df['osmid'][i]: node_df.loc[i][['y', 'x']].to_numpy() for i in range(node_df.shape[0])}
map_edge_id_to_u_v = edge_df[['u', 'v']].to_numpy()
num_edges = map_edge_id_to_u_v.shape[0]

haversines = {}
for e in tqdm(range(num_edges)):
	u, v = map_edge_id_to_u_v[e]
	coords1 = map_node_osm_to_coords[u]
	coords2 = map_node_osm_to_coords[v]
	h = haversine(coords1, coords2, unit = "km")
	haversines[e] = h


def get_traffic_features(filename = None, num_days = 1, train = False, num_components = 10, device = "cpu", find_interval = None):
	data = pickle.load(open(filename,"rb"))
	print("Loaded all data files")

	trip_lens = {}
	trip_speeds = {} #in km/hr
	counts = {}
	speed_sums = {}


	tot_sum = 0
	tot_count = 0

	intervals = set()
	for id, trip, (s,e) in tqdm(data):
		if args.dataset == "cityindia":
			e /= 1000 # the units were ms
			s /= 1000 # the units were ms
		
		if e == 0 or s ==0 :
			cprint("Timestamps missing, cannot compute edge speeds, try running without traffic", "red")
			raise SystemExit
		
		t = (e - s)/3600 # seconds to hours
			
		if t == 0:
			print("start and end time same")
			continue
			
		assert len(trip) >= 1, "trips not filtered properly"
		l = sum([haversines[e] for e in trip])
		speed = l/t
		trip_lens[id] = l
		trip_speeds[id] = speed

		interval = find_interval(s)
		intervals.add(interval)
		for e in trip:
			tup = (e,interval) 
			if tup not in counts:
				counts[tup] = 0
				speed_sums[tup] = 0

			counts[tup] += 1
			speed_sums[tup] += speed

			tot_count += 1
			tot_sum += speed

	max_day = max(d for d,_ in intervals)
	min_day = min(d for d,_ in intervals)


	print("assumption that no date coincides, unique dates")
	all_days = sorted(list(set((d for d,_ in intervals))))
	print("found min_day and max_day as {} and {}".format(min_day, max_day))
	all_intervals = [(d,hr) for hr in range(24) for d in all_days]

	avg_speed = tot_sum/tot_count
	print("avg speed across the network was found to be {} km/hr".format(round(avg_speed,2)))

	if not train:
		pca, chosen_edges, avg_speed = pickle.load(open(LEARNT_FEATURE_REP, "rb"))
		num_components = pca.n_components

	speeds = {}
	temp = set()
	for e in range(num_edges):
		for interval in all_intervals:
			k = (e, interval)
			if k in counts and counts[k] >= DATA_THRESHOLD:
				speeds[k] = speed_sums[k]/counts[k]
				temp.add(e)
			else:
				if k in counts:
					speeds[k] = (speed_sums[k] + (DATA_THRESHOLD - counts[k])*avg_speed)/DATA_THRESHOLD
				else:
					speeds[k] = avg_speed

	if train:
		chosen_edges = temp

	if len(chosen_edges) == 0:
		cprint("INSUFFICIENT DATA", "red")
		raise SystemExit

	cprint("number of chosen edges is {}".format(len(chosen_edges)), "green")

	data_array = np.empty((len(chosen_edges), len(all_intervals)), dtype = np.float32)
	chosen_edges = list(chosen_edges)

	for i,e in enumerate(chosen_edges):
		vec = np.array([speeds[(e,interval)] for interval in all_intervals], dtype = np.float32)
		data_array[i] = vec
	data_array = data_array.T

	if train:
		pca = PCA(n_components=num_components)
		pca.fit(data_array)
		print(pca.explained_variance_ratio_)
		print("retained {}% variance by creating just {} out of the original {} features"
			.format(round(sum(pca.explained_variance_ratio_*100), 2), num_components, data_array.shape[1]))
		pickle.dump((pca, chosen_edges, avg_speed),open(LEARNT_FEATURE_REP, "wb"))
		
	reduced = pca.transform(np.vstack((np.array([avg_speed for _ in range(data_array.shape[1])]),data_array)))
	return {e:torch.from_numpy(reduced[i]).float().to(device).reshape(1,-1) for i,e in enumerate(all_intervals)}

def fetch_traffic_features_stored(train_file = TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS,
									test_file = TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS, device = "cpu",
									find_interval = find_interval_1):

	store1 = get_traffic_features(filename = TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS, train = True, device = device, find_interval = find_interval)
	store2 = get_traffic_features(filename = VAL_TRIP_DATA_PICKLED_WITH_TIMESTAMPS, train = False, device = device, find_interval = find_interval)
	store3 = get_traffic_features(filename = TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS, train = False, device = device, find_interval = find_interval)
	try:
		assert len(set(store1.keys()).intersection(set(store2.keys()))) == 0, "problem with intervals"
		assert len(set(store1.keys()).intersection(set(store3.keys()))) == 0, "problem with intervals"
	except Exception as e:
		print("ignoring this exception for now")

	store1.update(store2)
	store1.update(store3)
	return store1

if __name__ == "__main__":
	store = fetch_traffic_features_stored(device = "cpu", find_interval = find_interval_2)
	all_intervals = list(store.keys())
	all_intervals.sort()
	forward_interval_map = {interval:index for index, interval in enumerate(all_intervals)}
	backward_interval_map = all_intervals
	traffic_matrix = torch.empty((len(all_intervals), 10))
	for i,interval in enumerate(all_intervals):
		traffic_matrix[i] = store[interval]
	traffic_matrix = traffic_matrix.float().numpy()

	print(traffic_matrix)
	perplexity = 10
	X_embedded = TSNE(n_components=2, perplexity = perplexity).fit_transform(traffic_matrix)

	plt.clf()
	plt.scatter(X_embedded[:,0], X_embedded[:,1])

	for i, interval in enumerate(all_intervals):
		plt.annotate(str(interval), (X_embedded[i][0], X_embedded[i][1]))

	plt.savefig("junk/tsne_actual_perplexity_{}.png".format(perplexity))
