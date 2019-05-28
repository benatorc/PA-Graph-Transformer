import os
import pickle
import pdb


class stats_tracker():
    def __init__(self, stat_names):
        self.stat_names = stat_names
        stats_agg = {}  # Keeps track of aggregate stats
        stats_norm = {}  # Keeps track of norm factor for each stat

        for name in stat_names:
            stats_agg[name] = 0
            stats_norm[name] = 0

        self.stats_agg = stats_agg
        self.stats_norm = stats_norm

    def add_stat(self, stat_name, val, norm=1):
        if stat_name not in self.stat_names:
            return
        self.stats_agg[stat_name] += val
        self.stats_norm[stat_name] += norm

    def get_stats(self):
        stats = {}
        for name in self.stat_names:
            if self.stats_norm[name] == 0:
                stats[name] = 0
            else:
                stats[name] = float(self.stats_agg[name]) / self.stats_norm[name]
        return stats


def read_splits(split_path):
    splits = {}

    split_file = open(split_path, 'r+')
    for line in split_file.readlines():
        data_type = line.strip().split(',')[0]
        split_indices = line.strip().split(',')[1:]
        split_indices = [int(x) for x in split_indices]
        splits[data_type] = split_indices
    return splits


def read_smiles_from_file(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles, label = line.strip().split(',')
        smiles_data.append((smiles, float(label)))
    data_file.close()
    return smiles_data


def read_smiles_multiclass(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles = line.strip().split(',')[0]
        labels = line.strip().split(',')[1:]
        labels = [float(x) for x in labels]

        smiles_data.append((smiles, labels))
    return smiles_data


def read_smiles_ring_data(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles = line.strip().split(',')[0]
        pair_labels = line.strip().split(',')[1:]

        atom_pairs = []
        labels = []
        for pair_label in pair_labels:
            pair_str, label_str = pair_label.split(':')
            pair = [int(x) for x in pair_str.split('-')]
            label = int(label_str)
            atom_pairs.append(pair)
            labels.append(label)
        smiles_data.append((smiles, (atom_pairs, labels)))
    return smiles_data


def read_smiles_from_dir(data_dir):
    smiles_data = {}
    for type in ['train', 'valid', 'test']:
        data_path = '%s/%s.txt' % (data_dir, type)

        data = read_smiles_from_file(data_path)
        smiles_data[type] = data
    return smiles_data


def load_shortest_paths(args):
    if args.model_type in ['graph_attn_net', 'transformer']:
        args.use_paths = True
        sp_file = '%s/shortest_paths.p' % args.data

        shortest_paths = pickle.load(open(sp_file, 'rb'))
        args.p_info = shortest_paths  # p info can also include rank information

        print('Shortest Paths loaded')


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def dict_to_pstr(dict, header_str, key_list=None):
    """Convert dictionary to a print-friendly string, sorted by key."""
    s = header_str

    if key_list is None:
        key_list = sorted(dict.keys())
    for k in key_list:
        s += ' %s: %.4f' % (k, dict[k])
    return s


def dict_to_dstr(dict, stat_names):
    """Convert dictionary to a csv-friendly string."""
    data_str = ''
    for name in stat_names:
        data_str += str(dict[name]) + ','
    return data_str[:-1]


def map_equiv(source_map, target_map):
    if source_map is None or target_map is None:
        return False

    if len(source_map) != len(target_map):
        return False

    for k, v in source_map.items():
        if k not in target_map:
            return False
        if v != target_map[k]:
            return False
    return True
