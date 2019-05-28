device=0
dataset=path_to_dataset

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mae -multi \
    -max_grad_norm 10 -batch_size 200 -num_epochs 300 \
    -output_dir output_test/qm8_conv_net -n_rounds 10  \
    -model_type conv_net -hidden_size 160 -lr 5e-4 -dropout 0.1

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mae -multi \
    -max_grad_norm 10 -batch_size 200 -num_epochs 300 \
    -output_dir output_test/qm8_transformer -n_rounds 10 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 5e-4 \
    -no_share -n_heads 1 -d_k 160 -dropout 0.1

CUDA_VISIBLE_DEVICES=$device python train/train_prop.py -cuda \
    -data $dataset -loss_type mae -multi \
    -max_grad_norm 10 -batch_size 200 -num_epochs 300 \
    -output_dir output_test/qm8_local -n_rounds 10 \
    -model_type transformer -hidden_size 160 \
    -p_embed -ring_embed -max_path_length 3 -lr 5e-4 \
    -mask_neigh -no_share -n_heads 1 -d_k 160 -dropout 0.1
