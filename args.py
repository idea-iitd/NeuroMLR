from argparse import ArgumentParser


def make_args():
	parser = ArgumentParser()
	parser.add_argument("-debug", action="store_true", dest = "debug")
	parser.add_argument("-learnable_slot_embeddings)", action="store_true", dest = "learnable_slot_embeddings")
	parser.add_argument("-only_train", action="store_false", dest = "test")
	parser.add_argument("-test", action="store_true", dest = "test")
	parser.add_argument("-haversine", action="store_true", dest="initialise_embeddings_haversine")
	parser.add_argument("-lipschitz", action="store_true", dest="initialise_embeddings_lipschitz")

	parser.add_argument("-fixed_embeddings", action="store_false", dest="trainable_embeddings")
	parser.add_argument("-memory_method", default = None, type = str, dest = "memory_method")
	parser.add_argument("-dataset", default = "beijing", type = str, dest = "dataset")
	parser.add_argument("-gnn", default = None, type = str, dest = "gnn")

	parser.add_argument("-with_correction", action="store_true", dest = "with_correction")

	parser.add_argument("-meta", action="store_true", dest = "meta")
	parser.add_argument("-traffic", action="store_true", dest = "traffic")

	parser.add_argument("-cache", action="store_true", dest = "cache")
	parser.add_argument("-remove_loops", action="store_true", dest = "remove_loops")
	parser.add_argument("-keep_loops_in_train", action="store_false", dest = "remove_loops_from_train")
	parser.add_argument("-keep_loops", action="store_false", dest = "remove_loops")
	parser.add_argument("-ignore_day", action="store_true", dest = "ignore_day")
	parser.add_argument("-all_split", action="store_true", dest = "all_split")
	parser.add_argument("-cpu", action="store_true", dest = "force_cpu")
	parser.add_argument("-roads_as_nodes", action="store_true", dest = "roads_as_nodes")

	parser.add_argument("-loss", default = "v2", type = str, dest = "loss")
	parser.add_argument("-gnn_layers", type=int, default=1, dest = "gnn_layers", help="number of layers in the GCN")
	parser.add_argument("-wait", type=int, default=0, dest = "wait", help="wait for these many epochs before evaluating")
	parser.add_argument("-eval_frequency", type=int, default=5, dest = "eval_frequency", help="evaluate afetr every these many epochs")
	
	parser.add_argument("-epochs", type=int, default=50, dest = "epochs", help="number of epochs")
	parser.add_argument("-gpu_index", type=int, default=0, dest = "gpu_index", help="which gpu")
	parser.add_argument("-batch_size", type=int, default=32, dest = "batch_size", help="Stochastic Gradient Descent mei batch ka size")
	parser.add_argument("-lr", type=float, default=0.001, dest = "lr", help="Learning Rate")
	parser.add_argument("-weight", type=float, default=1, dest = "weight", help="weight for one hot vectors")

	parser.add_argument("-end_dijkstra", action="store_true", dest="end_dijkstra")
	parser.add_argument("-no_end_dijkstra", action="store_false", dest="end_dijkstra")
	parser.add_argument("-initial_eval", action="store_true", dest="initial_eval")
	parser.add_argument("-check_script", action="store_true", dest="check_script")
	parser.add_argument("-attention", action="store_true", dest="attention")
	parser.add_argument("-one_hot_nodes", action="store_true", dest="one_hot_nodes")
	parser.add_argument("-one_hot_traffic", action="store_true", dest="one_hot_traffic")
	parser.add_argument("-unfiltered", action="store_true", dest="unfiltered")
	parser.add_argument("-filtered", action="store_false", dest="unfiltered")
	parser.add_argument("-intermediate_destinations", action="store_true", dest="intermediate_destinations")
	parser.add_argument("-end_destinations", action="store_false", dest="intermediate_destinations")
	parser.add_argument("-ignore_unknown_args", action="store_true", dest="ignore_unknown_args")
	parser.add_argument("-with_dijkstra", action="store_true", dest="with_dijkstra")

	parser.add_argument("-embedding_size", type=int, default=128, dest = "embedding_size")
	parser.add_argument("-hidden_size", type=int, default=256, dest = "hidden_size")
	parser.add_argument("-num_layers", type=int, default=3, dest = "num_layers")

	parser.add_argument("-linear_transform_initial", action="store_true", dest="linear_transform_initial")
	parser.add_argument("-percent_data", type=float, default=None, dest = "percent_data", help="percentage of training data to use")
	parser.add_argument("-result_file", default = None, type = str, dest = "result_file")
	parser.add_argument("-model_uid", default = '', type = str, dest = "model_uid")
	parser.add_argument("-num_heads", type=int, default=2, dest = "num_heads")
	parser.add_argument("-override_warnings", action="store_true", dest="override_warnings")
	parser.add_argument("-model_path", default = '', type = str, dest = "model_path")

	parser.set_defaults(override_warnings = False)
	parser.set_defaults(end_dijkstra = True)
	parser.set_defaults(linear_transform_initial = False)
	parser.set_defaults(with_dijkstra = False)
	parser.set_defaults(roads_as_nodes = False)
	parser.set_defaults(traffic = False)
	parser.set_defaults(meta = False)
	parser.set_defaults(with_correction = False)
	parser.set_defaults(force_cpu = False)
	parser.set_defaults(remove_loops = False)
	parser.set_defaults(remove_loops_from_train = True)
	parser.set_defaults(all_split = False)
	parser.set_defaults(intermediate_destinations = False)
	parser.set_defaults(ignore_unknown_args = False)
	parser.set_defaults(initial_eval = False)
	parser.set_defaults(debug = False)
	parser.set_defaults(attention = False)
	parser.set_defaults(unfiltered = True)
	parser.set_defaults(ignore_day = False)
	parser.set_defaults(test = True)
	parser.set_defaults(one_hot_nodes = False)
	parser.set_defaults(one_hot_traffic = False)
	parser.set_defaults(learnable_slot_embeddings = False)
	parser.set_defaults(initialise_embeddings_haversine = False)
	parser.set_defaults(initialise_embeddings_lipschitz = False)
	parser.set_defaults(check_script = False)
	
	parser.set_defaults(trainable_embeddings = True)
	parser.set_defaults(cache = False)

	
	args, unknown = parser.parse_known_args()
	if len(unknown)!= 0 and not args.ignore_unknown_args:
		print("some unrecognised arguments {}".format(unknown))
		raise SystemExit


	return args


