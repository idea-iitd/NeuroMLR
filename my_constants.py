from args import *
from termcolor import colored, cprint

#########################################################################
# please set this path according to the extracted folder
# PREFIX_PATH = None
PREFIX_PATH = "/home/qpp/data/preprocessed_data/"
#########################################################################

if PREFIX_PATH is None:
	cprint("\n\nPlease set the PREFIX_PATH after downloading and extracting the preprocessed data\n\n", "red")
	raise SystemExit

args = make_args()
PREFIX_PATH += "{}/".format(args.dataset)

def get_fname(percent_data, s):
	assert s[-4:] == '.pkl', colored('wrong file format', 'red')
	fname = s[:-4] + '_partial_{}.pkl'.format(percent_data)
	return fname

OUTDATED_DATASETS = []
PERCENTAGES = [1, 2, 5, 10, 20, 100]

EDGE_DATA = PREFIX_PATH + "map/edges.shp"
NODE_DATA = PREFIX_PATH + "map/nodes.shp"

TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_train_trips_all.pkl"

TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_test_trips_all.pkl"

VAL_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_validation_trips_all.pkl"

TRAIN_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_train_trips_small.pkl"
TEST_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_test_trips_small.pkl"
VAL_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_validation_trips_small.pkl"

assert not (args.check_script and args.percent_data is not None), colored('cannot take percent data with check script','red')
assert args.dataset not in OUTDATED_DATASETS or args.override_warnings, colored("these datasets are outdated - {}, are you sure you wanted to run this?".format(OUTDATED_DATASETS), 'red')

if args.check_script:
	TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = TRAIN_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS
	TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = TEST_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS
	VAL_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = VAL_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS
elif args.percent_data is not None:
	assert args.percent_data in PERCENTAGES, colored(
		"Choose a percentage value from the predefined ones - {}".format(PERCENTAGES), "red")
	assert args.dataset in ["cityindia", "harbin", "beijing"] , colored(
		"Small percentage of training data not created for this dataset", "red")

	TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = get_fname(args.percent_data, TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)

PICKLED_GRAPH = PREFIX_PATH + "map/graph_with_haversine.pkl"
LEARNT_FEATURE_REP = PREFIX_PATH + "pickled_pca_information_for_traffic_representation.pkl"
CRUCIAL_PAIRS = PREFIX_PATH + "crucial_pairs.pkl"
CACHED_DATA_FILE = PREFIX_PATH + "cached_data.pkl"
INITIALISED_EMBEDDINGS_PATH = "results/" 

model_path = args.model_path if args.model_path != '' else PREFIX_PATH + 'pretrained_models/'
MODEL_SAVE_PATH = model_path + 'model.pt'
MODEL_SUPPORT_PATH = model_path + 'model_support.pkl'