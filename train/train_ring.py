import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

from arguments import get_args, create_dirs
from train.train_base import train_model, test_model
from models.atom_predictor import AtomPredictor
from datasets.mol_dataset import get_loader
from graph.mol_graph import MolGraph
from utils import data_utils, train_utils
import pdb


def init_model(args, n_classes):
    atom_predictor = AtomPredictor(args, n_classes=n_classes)
    atom_predictor.to(args.device)

    optimizer = torch.optim.Adam(atom_predictor.parameters(), lr=args.lr)
    return atom_predictor, optimizer


def load_datasets(raw_data, split_idx, args, n_workers=5):
    data_splits = data_utils.read_splits('%s/split_%d.txt' % (args.data, split_idx))
    dataset_loaders = {}
    if not args.test_mode:
        dataset_loaders['train'] = get_loader(
            raw_data, data_splits['train'], args, shuffle=True, num_workers=n_workers)
        dataset_loaders['valid'] = get_loader(
            raw_data, data_splits['valid'], args, shuffle=False, num_workers=n_workers)
    dataset_loaders['test'] = get_loader(
        raw_data, data_splits['test'], args, shuffle=False, num_workers=n_workers)
    return dataset_loaders


def get_test_loader(raw_data, split_idx, args):
    data_splits = data_utils.read_splits('%s/split_%d.txt' % (args.data, split_idx))
    test_loader = get_loader(raw_data, data_splits['test'], args,
                             shuffle=False, num_workers=0, batch_size=1)
    return test_loader


def main():
    args = get_args()

    model_types = ['conv_net', 'conv_net_attn', 'transformer']
    assert args.model_type in model_types

    raw_data = data_utils.read_smiles_ring_data('%s/raw.csv' % args.data)

    atom_predictor, optimizer = init_model(args, args.n_classes)
    data_utils.load_shortest_paths(args)  # Shortest paths includes all splits

    agg_stats = ['loss', 'nei_score', 'acc', 'auc', 'gnorm', 'gnorm_clip']

    selection_stat = 'acc'
    select_higher = True

    if args.test_mode:
        dataset_loaders = load_datasets(raw_data, 0, args)
        test_model(
            dataset_loaders=dataset_loaders,
            model=atom_predictor,
            stat_names=agg_stats,
            train_func=run_epoch,
            args=args,)
        exit()

    all_stats = {}
    for name in agg_stats:
        all_stats[name] = []
    output_dir = args.output_dir
    all_model_paths = []

    for round_idx in range(args.n_rounds):
        dataset_loaders = load_datasets(raw_data, round_idx, args, n_workers=0)
        atom_predictor, optimizer = init_model(args, args.n_classes)

        cur_output_dir = '%s/run_%d' % (output_dir, round_idx)
        args.output_dir = cur_output_dir
        create_dirs(args, cur_output_dir)

        test_stats, best_model_path = train_model(
            dataset_loaders=dataset_loaders,
            model=atom_predictor,
            optimizer=optimizer,
            stat_names=agg_stats,
            selection_stat=selection_stat,
            train_func=run_epoch,
            args=args,
            select_higher=select_higher,)

        # Aggregate stats of interest
        for name in agg_stats:
            all_stats[name].append(test_stats[name])
        all_model_paths.append(best_model_path)

    # Write summary file
    summary_file = open('%s/summary.txt' % output_dir, 'w+')

    for name, stats_arr in all_stats.items():
        stats = np.array(stats_arr)
        mean, std = np.mean(stats), np.std(stats)
        stats_string = '%s: %s, mean: %.3f, std: %.3f' % (name, str(stats_arr), mean, std)
        print(stats_string)
        summary_file.write('%s\n' % stats_string)

    for model_path in all_model_paths:
        summary_file.write('%s\n' % model_path)

    summary_file.close()


def run_epoch(data_loader, model, optimizer, stat_names, args, mode,
              write_path=None):
    training = mode == 'train'
    atom_predictor = model
    atom_predictor.train() if training else atom_predictor.eval()

    if write_path is not None:
        write_file = open(write_path, 'w+')
    stats_tracker = data_utils.stats_tracker(stat_names)

    batch_split_idx = 0
    all_pred_logits, all_labels = [], []  # Used to compute Acc, AUC
    for batch_idx, batch_data in enumerate(tqdm.tqdm(data_loader, dynamic_ncols=True)):
        if training and batch_split_idx % args.batch_splits == 0:
            optimizer.zero_grad()
        batch_split_idx += 1

        smiles_list, labels_list, path_tuple = batch_data
        path_input, path_mask = path_tuple
        if args.use_paths:
            path_input = path_input.to(args.device)
            path_mask = path_mask.to(args.device)

        n_data = len(smiles_list)
        mol_graph = MolGraph(smiles_list, args, path_input, path_mask)
        atom_pairs_idx, labels = zip(*labels_list)

        pred_logits = atom_predictor(
            mol_graph, atom_pairs_idx, stats_tracker).squeeze(1)
        labels = [torch.tensor(x, device=args.device) for x in labels]
        labels = torch.cat(labels, dim=0)

        all_pred_logits.append(pred_logits)
        all_labels.append(labels)

        if args.n_classes > 1:
            pred_probs = nn.Softmax(dim=1)(pred_logits)
            loss = F.cross_entropy(input=pred_logits, target=labels)
        else:
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels.float())

        stats_tracker.add_stat('loss', loss.item() * n_data, n_data)
        loss = loss / args.batch_splits

        if write_path is not None:
            write_ring_output(
                write_file, smiles_list, atom_pairs_idx, labels, pred_probs, args.n_classes)

        if training:
            loss.backward()

            if batch_split_idx % args.batch_splits == 0:
                train_utils.backprop_grads(
                    atom_predictor, optimizer, stats_tracker, args)
                batch_split_idx = 0

    if training and batch_split_idx != 0:
        train_utils.backprop_grads(model, optimizer, stats_tracker, args)  # Any remaining

    all_pred_logits = torch.cat(all_pred_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if args.n_classes > 1:
        pred_probs = nn.Softmax(dim=1)(all_pred_logits).detach().cpu().numpy()
    else:
        pred_probs = nn.Sigmoid()(all_pred_logits).detach().cpu().numpy()
    all_labels = all_labels.detach().cpu().numpy()
    acc = train_utils.compute_acc(pred_probs, all_labels, args.n_classes)

    if args.n_classes > 1:
        auc = 0
    else:
        auc = train_utils.compute_auc(pred_probs, all_labels)
    stats_tracker.add_stat('acc', acc, 1)
    stats_tracker.add_stat('auc', auc, 1)

    if write_path is not None:
        write_file.close()
    return stats_tracker.get_stats()


def write_ring_output(write_file, smiles_list, atom_pair_idx, labels, preds, n_classes=1):
    idx = 0
    for mol_idx, atom_pairs in enumerate(atom_pair_idx):
        write_str = '%s' % smiles_list[mol_idx]

        for pair_idx, atom_pair in enumerate(atom_pairs):
            label = labels[idx].item()
            if n_classes > 1:
                pred = torch.argmax(preds[idx]).item()
                pred_prob = preds[idx, pred].item()
            else:
                pred_prob = preds[idx].item()
                pred = int(pred_prob > 0.5)
            is_wrong = pred != label
            wrong_str = '***' if is_wrong else ''
            write_str += ',%s%d-%d:%d-%.3f' % (
                wrong_str, atom_pair[0], atom_pair[1], labels[idx].item(), pred_prob)
            idx += 1
        write_file.write('%s\n' % write_str)


if __name__ == '__main__':
    main()
