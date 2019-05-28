import torch
import argparse
import utils.data_utils as utils


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', action='store_true', default=False,
                        help='Whether or not to use GPU.')
    parser.add_argument('-data', type=str, default='data/clean',
                        help='Input data directory.')
    parser.add_argument('-output_dir', type=str, default='output/test',
                        help='The output directory.')

    parser.add_argument('-test_mode', action='store_true', default=False,
                        help='Activates test mode for debugging purposes.')
    parser.add_argument('-test_model', type=str, default='',
                        help='The model path used for testing')
    parser.add_argument('-data_split', type=int, default=0,
                        help='Test split to run model on')
    parser.add_argument('-pretrain_model', type=str, default='',
                        help='The model path used for pretrained model')
    parser.add_argument('-multi', action='store_true', default=False,
                        help='Indicator for being multiclass training.')
    parser.add_argument('-n_classes', type=int, default=1,
                        help='Helper variable for number of classes')

    parser.add_argument('-model_type', type=str, default='',
                        help='The graph model type to use')
    parser.add_argument('-loss_type', type=str, default='mse',
                        help='The loss type for the dataset')
    parser.add_argument('-agg_func', type=str, default='sum',
                        help='Agg function, sum or mean')
    parser.add_argument('-self_attn', action='store_true', default=False,
                        help='Whether or not to include self in attn')

    parser.add_argument('-n_rounds', type=int, default=5,
                        help='Number of times to run a model')
    parser.add_argument('-num_epochs', type=int, default=10,
                        help='Number of epochs to train model.')
    parser.add_argument('-batch_size', type=int, default=24,
                        help='Number of examples per batch.')
    parser.add_argument('-batch_splits', type=int, default=1,
                        help='Used to aggregate batches')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='The default learning rate for the optimizer.')
    parser.add_argument('-dropout', type=float, default=0.0,
                        help='The dropout probability for the model.')
    parser.add_argument('-max_grad_norm', type=float, default=1.0,
                        help='The maximum gradient norm allowed')
    parser.add_argument('-hidden_size', type=int, default=200,
                        help='The number of hidden units for the model.')
    parser.add_argument('-depth', type=int, default=5,
                        help='The depth of the net.')

    # Transformer arguments
    parser.add_argument('-n_heads', type=int, default=1,
                        help='Number of heads in multihead attention.')
    parser.add_argument('-d_k', type=int, default=32,
                        help='The size of each indvidiual attention head')

    # Path Arguments
    parser.add_argument('-max_path_length', type=int, default=5,
                        help='The max path length to consider')
    parser.add_argument('-p_embed', action='store_true', default=False,
                        help='use distance position or not')
    parser.add_argument('-ring_embed', action='store_true', default=False,
                        help='use ring in path info')
    parser.add_argument('-no_truncate', action='store_true', default=False,
                        help='Whether or not to truncate atom paths.')

    # Graph Attn Network Arguments
    parser.add_argument('-no_share', action='store_true', default=False,
                        help='Whether to deactivate weight sharing for gcn')
    parser.add_argument('-mask_neigh', action='store_true', default=False,
                        help='Whether or not to mask outside neighborhood')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if args.cuda else 'cpu')
    print('Using device %s' % (args.device))

    if not args.test_mode:
        create_dirs(args, args.output_dir)
        write_args(args)

    args.use_paths = False
    return args


def create_dirs(args, output_dir):
    utils.create_dir_if_not_exists(output_dir)
    args.model_dir = '%s/models' % output_dir
    args.result_dir = '%s/results' % output_dir

    for dir in [args.model_dir, args.result_dir]:
        utils.create_dir_if_not_exists(dir)


def write_args(args):
    params_file = open('%s/params.txt' % args.output_dir, 'w+')
    for attr, value in sorted(args.__dict__.items()):
        params_file.write("%s=%s\n" % (attr, value))
    params_file.close()
