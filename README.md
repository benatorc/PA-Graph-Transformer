# Path-Augmented Graph Transformer Network


This is the github repo for the paper "Path-Augmented Graph Transformer Network"

All data (and splits used for experiments are under data.zip)


These are the require packages and set up for a conda environment (can be slightly different depending on system).

```
conda create -c rdkit -n prop_predictor rdkit
source activate prop_predictor
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install scikit-learn tqdm
```

Add the repo to PYTHONPATH:

```
export PYTHONPATH=path_to_repo:
```

To compute the shortest paths, run the following:
```
python preprocess/shortest_paths.py -data_dir path_to_data
```

To run the (transformer) model:
```
dataset=path_to_dataset
python train/train_prop.py -cuda \
    -data $dataset -loss_type mse \
    -max_grad_norm 10 -batch_size 50 -num_epochs 100 \
	-output_dir output_test/sol_transformer -n_rounds 10 \
	-model_type transformer -hidden_size 160 \
	-p_embed -ring_embed -max_path_length 3 -lr 5e-4 \
	-no_share -n_heads 2 -d_k 80 -dropout 0.2
```

To run the local model, add the option:
```
-mask_neigh
```

To run the conv net model:
```
dataset=path_to_dataset
python train/train_prop.py -cuda \
    -data $dataset -loss_type mse \
    -max_grad_norm 10 -batch_size 50 -num_epochs 100 \
    -output_dir output_test/sol_conv_net -n_rounds 10  \
    -model_type conv_net -hidden_size 160 -lr 5e-4 -dropout 0.2
```


All the scripts used to generate the results in the paper are under the scripts directory.
