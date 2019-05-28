def write_props(write_file, smiles_list, labels_list, preds_list):
    for idx in range(len(smiles_list)):
        smiles = smiles_list[idx]
        label = labels_list[idx]
        pred = preds_list[idx]

        write_file.write('%s,%s,%s\n' % (smiles, str(label), str(pred)))
