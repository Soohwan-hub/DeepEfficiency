import torch
import esm
import re
import numpy as np
import data.utils as utils


from transformers import AutoTokenizer, AutoModel
from transformers import T5EncoderModel, T5Tokenizer
from unimol_tools import UniMolRepr

def enzyme_encoder(train, model_name, batch_size = 1):
    r"""
    recieve training set and will encode the enzyme info utilizing ESM 2 and return embedding
    """
    if model_name == "ProtT5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        sequences = train["Sequence"].tolist()
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i: i+batch_size]
            ids = tokenizer(
                batch_seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt"
            )
            input_ids = ids["input_ids"].to(device)
            attention_mask = ids["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state

                mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask

                sum_embed = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)

                mean_pooled_embed = (sum_embed / sum_mask).cpu().numpy()
                all_embeddings.append(mean_pooled_embed)
        return np.vstack(all_embeddings)

    if model_name == "ESM 3":
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ESM3.from_pretrained("esm3-sm-open-v1").to(device)
        model.eval()

        sequences = train["Sequence"].tolist()
        all_embeddings = []
        for seq in sequences:
            protein = ESMProtein(sequence=seq)
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                emb = logits_output.embeddings
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                emb = np.asarray(emb)
                if emb.ndim == 2:
                    emb = emb.mean(axis=0)
                all_embeddings.append(emb)
            
            del protein_tensor, logits_output
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        return np.vstack(all_embeddings)

    models = {
        "ESM 2 650M": esm.pretrained.esm2_t33_650M_UR50D(),
        "ESM 2 3B": esm.pretrained.esm2_t36_3B_UR50D(),
    }
    model, alphabet = models[model_name]
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

    

def substrate_encoder(train, model_name, batch_size = 16):
    r"""
    Receives training dataframe, encodes substrate SMILES in mini-batches 
    utilizing ChemBERTa-77M-MTR, applies masked mean pooling, and returns a Numpy matrix.
    """
    if model_name == "Uni-Mol":
        smiles_list = train["SMILES"].tolist()

        repr_model = UniMolRepr(
            data_type="molecule",
            batch_size =batch_size,
            remove_hs=False,
            model_name="unimolv1",
            use_ddp=False,
            use_gpu=1
        )

        out = repr_model.get_repr(smiles_list, return_atomic_reprs=False)
        
        if isinstance(out, dict):
            cls_repr = out.get("cls_repr", out.get("cls_reprs"))
        elif isinstance(out, list):
            if not out:
                raise ValueError("Uni-Mol returned an empty representation list.")
            if isinstance(out[0], dict):
                cls_repr = [row.get("cls_repr", row.get("cls_reprs")) for row in out]
            else:
                cls_repr = out
        else:
            raise TypeError(f"Unexpected Uni-Mol output type: {type(out)}")

        if cls_repr is None:
            raise ValueError("Uni-Mol output did not include cls_repr.")
        return np.array(cls_repr)

    models = {"ChemBERTa-MTR": "DeepChem/ChemBERTa-77M-MTR", "ChemBERTa-MLM": "DeepChem/ChemBERTa-77M-MLM"}
    model_name = models[model_name]
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

def concat_encoder(train, enzyme_model_name, substrate_model_name, enzyme_batch_size=1, substrate_batch_size=64):
    enzyme_embed = enzyme_encoder(train, enzyme_model_name, enzyme_batch_size)
    substrate_embed = substrate_encoder(train, substrate_model_name, substrate_batch_size)
    assay_conditions = train[["PH", "Temperature"]].to_numpy(dtype=float)
    return np.concatenate([enzyme_embed, substrate_embed, assay_conditions], axis=1)

def main():
    data_path = "data/data_KCATKM.csv"
    train, val, test = utils.split_data(data_path=data_path)