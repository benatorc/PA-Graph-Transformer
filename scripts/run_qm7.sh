device=0
dataset=path_to_dataset

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mae \
    -max_grad_norm 10 -batch_size 50 -num_epochs 100 \
    -output_dir output_test/qm7_conv_net -n_rounds 10  \
    -model_type conv_net -hidden_size 300 -lr 1e-3 -dropout 0.2

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mae \
    -max_grad_norm 10 -batch_size 50 -num_epochs 100 \
    -output_dir output_test/qm7_transformer -n_rounds 10 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 1e-3 \
    -no_share -n_heads 1 -d_k 160 -dropout 0.2

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mae \
    -max_grad_norm 10 -batch_size 50 -num_epochs 100 \
    -output_dir output_test/qm7_local -n_rounds 10 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 1e-3 \
    -mask_neigh -no_share -n_heads 1 -d_k 160 -dropout 0.2
