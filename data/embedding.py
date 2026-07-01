import torch
import esm
import numpy as np
import data.utils as utils


from transformers import AutoTokenizer, AutoModel

def enzyme_encoder(train, batch_size = 2):
    r"""
    recieve training set and will encode the enzyme info utilizing ESM 2 and return embedding
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sequences = train["Sequence"].tolist()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_data = [(str(idx), seq) for idx, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
        
        for k, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            embeddings = token_representations[k, 1:seq_len + 1]
            mean_pooled_vec = embeddings.mean(dim=0).cpu().numpy()
            all_embeddings.append(mean_pooled_vec)
    return np.array(all_embeddings)

    

def substrate_encoder(train, batch_size = 16):
    r"""
    Receives training dataframe, encodes substrate SMILES in mini-batches 
    utilizing ChemBERTa-77M-MTR, applies masked mean pooling, and returns a Numpy matrix.
    """
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    smiles_list = train["SMILES"].tolist()
    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        encoded_input = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            token_embeddings = model_output.last_hidden_state
        
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1)
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
        all_embeddings.append(mean_pooled) 
    
    return np.vstack(all_embeddings)

def concat_encoder(train, enzyme_batch_size=2, substrate_batch_size=64):
    enzyme_embed = enzyme_encoder(train, enzyme_batch_size)
    substrate_embed = substrate_encoder(train, substrate_batch_size)
    assay_conditions = train[["PH", "Temperature"]].to_numpy(dtype=float)
    return np.concatenate([enzyme_embed, substrate_embed, assay_conditions], axis=1)

def main():
    data_path = "data/data_KCATKM.csv"
    train, val, test = utils.split_data(data_path=data_path)