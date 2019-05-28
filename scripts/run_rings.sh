device=0
dataset=path_to_dataset

CUDA_VISIBLE_DEVICES=$device python train/train_ring.py -cuda \
    -data $dataset -loss_type ce \
    -max_grad_norm 10 -batch_size 50 -num_epochs 50 \
    -output_dir output_test/ring_conv_net -n_rounds 5  \
    -model_type conv_net -hidden_size 160 -lr 1e-3

CUDA_VISIBLE_DEVICES=$device python train/train_ring.py -cuda \
    -data $dataset -loss_type ce \
    -max_grad_norm 10 -batch_size 25 -batch_splits 2 -num_epochs 50 \
    -output_dir output_test/ring_transformer -n_rounds 5 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 1e-3 \
    -no_share -n_heads 1 -d_k 160
