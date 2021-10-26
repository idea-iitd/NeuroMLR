<!-- README.md -->

# NeuroMLR: Robust & Reliable Route Recommendation on Road Networks
This repository is the official implementation of __NeuroMLR: Robust & Reliable Route Recommendation on Road Networks__.

## Introduction
Predicting the most likely route from a source location to a destination is a core functionality in mapping services. Although the problem has been studied in the literature, two key limitations remain to be addressed. First, a significant portion of the routes recommended by existing methods fail to reach the destination. Second, existing techniques are transductive in nature; hence, they fail to recommend routes if unseen roads are encountered at inference time. We address these limitations through an inductive algorithm called NEUROMLR. NEUROMLR learns a generative model from historical trajectories by conditioning on three explanatory factors: the current location, the destination, and real-time traffic conditions. The conditional distributions are learned through a novel combination of Lipschitz embeddings with Graph Convolutional Networks (GCN) on historical trajectories.

## Requirements

### Dependencies
The code has been tested for Python version 3.8.10 and CUDA 10.2. We recommend that you use the same. 

To create a virtual environment using conda, 
```bash
conda create -n ENV_NAME python=3.8.10
conda activate ENV_NAME
```

All dependencies can be installed by running the following commands - 

```bash
pip install -r requirements.txt
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-geometric
```

### Data
Download the [preprocessed data](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view?usp=sharing) and unzip the downloaded .zip file.  

Set the PREFIX_PATH variable in `my_constants.py` as the path to this extracted folder.

For each city (Chengdu, Harbin, Porto, Beijing, CityIndia), there are two types of data:

#### 1. Mapmatched pickled trajectories

Stored as a python pickled list of tuples, where each tuple is of the form (trip_id, trip, time_info). Here each trip is a list of edge identifiers.


#### 2. OSM map data
	
In the map folder, there are the following files-

1. `nodes.shp` : Contains OSM node information (global node id mapped to (latitude, longitude)) 
2. `edges.shp` : Contains network connectivity information (global edge id mapped to corresponding node ids)
3. `graph_with_haversine.pkl` : Pickled NetworkX graph corresponding to the OSM data  


## Training
After setting PREFIX_PATH in the `my_constants.py` file, the training script can be run directly as follows- 
```bash
python train.py -dataset beijing -gnn GCN -lipschitz 
``` 
Other functionality can be toggled by adding them as arguments, for example,

```bash
python train.py -dataset DATASET -gpu_index GPU_ID -eval_frequency EVALUATION_PERIOD_IN_EPOCHS -epochs NUM_EPOCHS 
python train.py -traffic
python train.py -check_script
python train.py -cpu

```

Brief description of other arguments/functionality - 

<!-- - _-check_script_: to run on partial subset of train_data, as a sanity test
- _-cpu_: forces computation on a cpu instead of the available gpu
- _-gnn_: can choose between a GCN or a GAT
- _-gnn_layers_: number of layers for the graph neural network used
- _-epochs_: number of epochs to train for
- _-percent_data_: percentage data used for training
- _-fixed_embeddings_: to make the embeddings static, they aren't learnt as parameters of the network
- _-embedding_size_: the dimension of embeddings used
- _-hidden_size_: hidden dimension for the MLP 
- _-traffic_: to toggle the attention module
- _-attention_: to toggle the attention module -->


| Argument  | Functionality |
| ------------- |-------------|
| *-check_script* | to run on a fixed subset of train_data, as a sanity test |
| _-cpu_ | forces computation on a cpu instead of the available gpu |
| _-gnn_ | can choose between a GCN or a GAT |
| _-gnn_layers_ | number of layers for the graph neural network used |
| _-epochs_ | number of epochs to train for |
| _-percent\_data_ | percentage data used for training |
| _-fixed_embeddings_ | to make the embeddings static, they aren't learnt as parameters of the network |
| _-embedding_size_ | the dimension of embeddings used |
| _-hidden_size_ | hidden dimension for the MLP  |
| _-traffic_ | to toggle the attention module |

For exact details about the expected format and possible inputs please refer to the `args.py` and `my_constants.py` files. 

## Evaluation
The training code generates logs for evaluation. To evaluate any pretrained model, run
```bash
python eval.py -dataset DATASET -model_path MODEL_PATH
```
There should be two files under MODEL_PATH, namely `model.pt` and `model_support.pkl` (refer to the function `save_model()` defined in `train.py` to understand these files).


## Pre-trained Models
You can find the pretrained models in the same [zip](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view?usp=sharing) as preprocessed data. To evaluate the models, set PREFIX_PATH in the _my\_constants.py_ file and run
```bash
python eval.py -dataset DATASET
```

## Results

We present the performance results of both versions of NeuroMLR across five datasets.

#### NeuroMLR-Greedy

| Dataset | Precision(%) | Recall(%) | Reachability(%) | Reachability distance (km) | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Beijing | 75.6 | 74.5 | 99.1 | 0.01 |
| Chengdu | 86.1 | 83.8 | 99.9 | 0.0002 |
| CityIndia | 74.3 | 70.1 | 96.1 | 0.03 |
| Harbin | 59.6 | 48.6 | 99.1 | 0.02 |
| Porto | 77.3 | 70.7 | 99.6 | 0.001 |

#### NeuroMLR-Dijkstra
Since NeuroMLR-Dijkstra guarantees reachability, the reachability metrics are not relevant here.

| Dataset | Precision(%) | Recall(%) |
| ------------- | ------------- | ------------- |
| Beijing | 77.9 | 76.5 |
| Chengdu | 86.7 | 84.2 |
| CityIndia | 77.9 | 73.1 |
| Harbin | 66.1 | 49.6 |
| Porto | 79.2 | 70.9 |

## Contributing
If you'd like to contribute, open an issue on this GitHub repository. All contributions are welcome! 
