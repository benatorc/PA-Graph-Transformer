device=0
dataset=path_to_dataset

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py \
    -cuda -data $dataset -loss_type mse \
    -batch_size 50 -output_dir output_test/lipo_conv_net -n_rounds 10 \
    -model_type conv_net -no_share -dropout 0.2 \
    -max_grad_norm 10 -hidden_size 160 -lr 5e-4 -num_epochs 100

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mse \
    -max_grad_norm 10 -batch_size 25 -batch_splits 2 -num_epochs 150 \
    -output_dir output_test/lipo_transformer -n_rounds 10 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 5e-4 \
    -no_share -n_heads 1 -d_k 160 -dropout 0.2

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mse \
    -max_grad_norm 10 -batch_size 25 -batch_splits 2 -num_epochs 150 \
    -output_dir output_test/lipo_local -n_rounds 10 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 5e-4 \
    -mask_neigh -no_share -n_heads 1 -d_k 160 -dropout 0.2
