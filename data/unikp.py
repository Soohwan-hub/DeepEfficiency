import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
# build_vocab, pretrain_trfm, utils packages are from SMILES Transformer
from transformers import T5EncoderModel, T5Tokenizer
# transformers package is from ProtTrans
import re
import gc
import os
import numpy as np
import pandas as pd
import pickle
import math
import data.utils as utils
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def _resolve_unikp_dir(unikp_dir=None):
    if unikp_dir:
        return os.path.abspath(os.path.expanduser(unikp_dir))
    env_unikp_dir = os.environ.get("UNIKP_DIR")
    if env_unikp_dir:
        return os.path.abspath(os.path.expanduser(env_unikp_dir))
    return os.path.abspath(os.path.expanduser("~/UniKP"))


def smiles_to_vec(Smiles, unikp_dir):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab_path = os.path.join(unikp_dir, "vocab.pkl")
    trfm_path = os.path.join(unikp_dir, "trfm_12_23000.pkl")
    vocab = WordVocab.load_vocab(vocab_path)
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    
    # UniKP's SMILES transformer calls `.numpy()` internally, so keep it on CPU.
    trfm_device = torch.device("cpu")
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(trfm_path, map_location=trfm_device))
    trfm.to(trfm_device)
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid).to(trfm_device))
    if isinstance(X, torch.Tensor):
        return X.cpu().numpy()
    return np.asarray(X)


def Seq_to_vec(Sequence, unikp_dir):
    sequence_list = list(Sequence)
    for i in range(len(sequence_list)):
        if len(sequence_list[i]) > 1000:
            sequence_list[i] = sequence_list[i][:500] + sequence_list[i][-500:]
    sequences_Example = []
    for i in range(len(sequence_list)):
        zj = ''
        for j in range(len(sequence_list[i]) - 1):
            zj += sequence_list[i][j] + ' '
        zj += sequence_list[i][-1]
        sequences_Example.append(zj)
    ###### you should place downloaded model into this directory.
    prot_t5_model_id = "Rostlab/prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(prot_t5_model_id, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(prot_t5_model_id)
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize

def UniKP(
        split_dir=None,
        unikp_dir=None
    ):
    unikp_dir = _resolve_unikp_dir(unikp_dir)
    train, val, test = utils.load_data_splits(split_dir)

    sequences = test["Sequence"]
    Smiles = test["SMILES"]
    seq_vec = Seq_to_vec(sequences, unikp_dir)
    smiles_vec = smiles_to_vec(Smiles, unikp_dir)
    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)

    ###### you should place downloaded model into this directory.
    model_path = os.path.join(unikp_dir, "UniKP for kcat_Km.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    Pre_label = model.predict(fused_vector)
    mse = mean_squared_error(test["Log10_value"], Pre_label)
    r2 = r2_score(test["Log10_value"], Pre_label)
    pearson, _ = pearsonr(test["Log10_value"], Pre_label)
    res = pd.DataFrame({'sequences': sequences, 'Smiles': Smiles, 'Pre_label': Pre_label})
    res.to_csv('Kinetic_parameters_predicted_label.csv')

    metrics = pd.DataFrame([{
        "model": "UniKP",
        "mse": mse,
        "r2": r2,
        "pearson": pearson
    }])
    metrics.to_csv("Kinetic_parameters_metrics.csv", index=False)
    
if __name__ == '__main__':
    UniKP()