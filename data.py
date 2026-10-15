import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from collections import Counter
from torch_geometric.loader import DataLoader
import esm
import pickle
from pyteomics import fasta

from tqdm import tqdm
from util import *

from Bio.PDB import PDBParser
from scipy import spatial
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
blosum62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]  # V
}
hydrophobicity = {
    'A': [100, 13, 60, 33, 98, 64, 39, 96, 71, 91, 93, 35, 89, 87, 89, 94, 98, 98, 94, 94],
    'R': [13, 100, 53, 81, 11, 49, 74, 17, 42, 4, 6, 78, 2, 0, 24, 19, 16, 11, 19, 7],
    'N': [60, 53, 100, 73, 58, 96, 79, 64, 89, 51, 52, 75, 49, 47, 71, 66, 63, 58, 66, 54],
    'D': [33, 81, 73, 100, 30, 68, 94, 36, 61, 23, 25, 98, 21, 19, 44, 39, 35, 31, 38, 26],
    'C': [98, 11, 58, 30, 100, 62, 36, 94, 69, 93, 95, 33, 91, 89, 86, 91, 95, 99, 92, 96],
    'Q': [64, 49, 96, 68, 62, 100, 74, 68, 93, 55, 57, 71, 53, 51, 76, 71, 67, 63, 70, 58],
    'E': [39, 74, 79, 94, 36, 74, 100, 43, 68, 29, 31, 96, 27, 26, 50, 45, 41, 37, 44, 33],
    'G': [96, 17, 64, 36, 94, 68, 43, 100, 75, 87, 89, 39, 85, 83, 93, 98, 99, 94, 98, 90],
    'H': [71, 42, 89, 61, 69, 93, 68, 75, 100, 62, 64, 64, 60, 58, 83, 78, 74, 69, 77, 65],
    'I': [91, 4, 51, 23, 93, 55, 29, 87, 62, 100, 98, 26, 98, 96, 79, 84, 88, 93, 85, 97],
    'L': [93, 6, 52, 25, 95, 57, 31, 89, 64, 98, 100, 27, 96, 94, 81, 86, 90, 94, 87, 99],
    'K': [35, 78, 75, 98, 33, 71, 96, 39, 64, 26, 27, 100, 24, 22, 46, 41, 38, 33, 41, 29],
    'M': [89, 2, 49, 21, 91, 53, 27, 85, 60, 98, 96, 24, 100, 98, 78, 83, 86, 91, 83, 95],
    'F': [87, 0, 47, 19, 89, 51, 26, 83, 58, 96, 94, 22, 98, 100, 76, 81, 84, 89, 81, 93],
    'P': [89, 24, 71, 44, 86, 76, 50, 93, 83, 79, 81, 46, 78, 76, 100, 95, 91, 87, 94, 83],
    'S': [94, 19, 66, 39, 91, 71, 45, 98, 78, 84, 86, 41, 83, 81, 95, 100, 96, 92, 99, 88],
    'T': [98, 16, 63, 35, 95, 67, 41, 99, 74, 88, 90, 38, 86, 84, 91, 96, 100, 96, 97, 91],
    'W': [98, 11, 58, 31, 99, 63, 37, 94, 69, 93, 94, 33, 91, 89, 87, 92, 96, 100, 93, 96],
    'Y': [94, 19, 66, 38, 92, 70, 44, 98, 77, 85, 87, 41, 83, 81, 94, 99, 97, 93, 100, 88],
    'V': [94, 7, 54, 26, 96, 58, 33, 90, 65, 97, 99, 29, 95, 93, 83, 88, 91, 96, 88, 100]
}
PAM120 = {
    'A': [3, -3, 0, 0, -3, -1, 0, 1, -3, -1, -3, -2, -2, -4, 1, 1, 1, -7, -4, 0],  # A
    'R': [-3, 6, -1, -3, -4, 1, -3, -4, 1, -2, -4, 2, -1, -4, -1, -1, -2, 1, -6, -3],  # R
    'N': [0, -1, 4, 2, -5, 0, 1, 0, 2, -2, -4, 1, -3, -4, -2, 1, 0, -5, -2, -3],  # N
    'D': [0, -3, 2, 5, -7, 1, 3, 0, 0, -3, -5, -1, -4, -7, -2, 0, -1, -8, -5, -3],  # D
    'C': [-3, -4, -5, -7, 9, -7, -7, -5, -4, -3, -7, -7, -6, -6, -3, -1, -3, -8, -1, -2],  # C
    'Q': [-1, 1, 0, 1, -7, 6, 2, -3, 3, -3, -2, 0, -1, -6, 0, -2, -2, -6, -5, -3],  # Q
    'E': [0, -3, 1, 3, -7, 2, 5, -1, -1, -3, -4, -1, -4, -6, -1, -1, -2, -8, -4, -3],  # E
    'G': [1, -4, 0, 0, -5, -3, -1, 5, -4, -4, -5, -3, -4, -5, -2, 1, -1, -8, -6, -2],  # G
    'H': [-3, 1, 2, 0, -4, 3, -1, -4, 7, -4, -3, -2, -4, -2, -1, -2, -3, -5, -1, -3],  # H
    'I': [-1, -2, -2, -3, -3, -3, -3, -4, -4, 6, 1, -2, 1, 0, -3, -2, 0, -7, -2, 3],  # I
    'L': [-3, -4, -4, -5, -7, -2, -4, -5, -3, 1, 5, -4, 3, 0, -3, -4, -3, -5, -3, 1],  # L
    'K': [-2, 2, 1, -1, -7, 0, -1, -3, -2, -2, -4, 5, 0, -6, -2, -1, -1, -5, -6, -4],  # K
    'M': [-2, -1, -3, -4, -6, -1, -4, -4, -4, 1, 3, 0, 8, -1, -3, -2, -1, -7, -4, 1],  # M
    'F': [-4, -4, -4, -7, -6, -6, -6, -5, -2, 0, 0, -6, -1, 8, -5, -3, -4, -1, 4, -3],  # F
    'P': [1, -1, -2, -2, -3, 0, -1, -2, -1, -3, -3, -2, -3, -5, 6, 1, -1, -7, -6, -2],  # P
    'S': [1, -1, 1, 0, -1, -2, -1, 1, -2, -2, -4, -1, -2, -3, 1, 3, 2, -2, -3, -2],  # S
    'T': [1, -2, 0, -1, -3, -2, -2, -1, -3, 0, -3, -1, -1, -4, -1, 2, 4, -6, -3, 0],  # T
    'W': [-7, 1, -5, -8, -8, -6, -8, -8, -5, -7, -5, -5, -7, -1, -7, -2, -6, 12, -1, -8],  # W
    'Y': [-4, -6, -2, -5, -1, -5, -4, -6, -1, -2, -3, -6, -4, 4, -6, -3, -3, -1, 8, -3],  # Y
    'V': [0, -3, -3, -3, -2, -3, -3, -2, -3, 3, 1, -4, 1, -3, -2, -2, 0, -8, -3, 5]  # V
}
def get_pam_embedding_from_peptide(fasta):
        """获取PAM120嵌入"""
        embedding = []
        for aa in fasta:
            if aa in PAM120:
                embedding.append(PAM120[aa])
            else:
                embedding.append(PAM120['A'])
        return np.array(embedding)
def AAC(sequence):
	# AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	AA = 'ACDEFGHIKLMNPQRSTVWY'

	count = Counter(sequence)
	for key in count:
		count[key] = count[key]/len(sequence)
	code = []
	for aa in AA:
		code.append(count[aa])
	return code
def AAC_encoding(seq_list):
    features = []

    for seq in seq_list:
        features.append(AAC(seq))

    return np.array(features, dtype=np.float32)

BLOSUM_DIM = 20
BLOSUM_SCALE = 11.0

def blosum62_features(seq):
    return np.asarray(
        [blosum62.get(aa.upper(), [0] * BLOSUM_DIM) for aa in seq],
        dtype=np.float32,
    ) / BLOSUM_SCALE
    
def blosum62_encoding(seq_list):
    return [blosum62_features(seq) for seq in seq_list]

def aaindex_ecoding(seqs):
    aaindex_path = (
        "../datasets/aaindex_acp.csv"
    )

    aaindex_df = pd.read_csv(
        aaindex_path,
        index_col="Description"
    ).astype(np.float32)

    row_mean = aaindex_df.mean(axis=1)
    row_std = aaindex_df.std(axis=1, ddof=0).replace(0, 1.0)

    normalized_df = aaindex_df.sub(
        row_mean,
        axis=0
    ).div(
        row_std,
        axis=0
    )

    aaindex_dict = {
        aa: normalized_df[aa].to_numpy(dtype=np.float32)
        for aa in normalized_df.columns
    }

    features = []

    feature_dim = normalized_df.shape[0]

    for seq in seqs:
        sequence_features = [
            aaindex_dict.get(
                aa.upper(),
                np.zeros(feature_dim, dtype=np.float32)
            )
            for aa in seq
        ]

        features.append(
            np.asarray(sequence_features, dtype=np.float32)
        )

    return features

def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of peptides.
    """
    d = 20
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return res

def BINARY(fastas):
	AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in range(1, len(fastas[0][1]) * 20 + 1):
		header.append('BINARY.F'+str(i))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for aa in sequence:
			if aa == '-':
				code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				continue
			for aa1 in AA:
				tag = 1 if aa == aa1 else 0
				code.append(tag)
		encodings.append(code)
	return encodings


def esm_embeddings(dataset):
	model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
	#model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
	batch_converter = alphabet.get_batch_converter()
	model.eval()
	device = torch.device("cuda")
	model = model.to(device)
	
	layer = 6
	#dataset = dataset.sort_values(lambda x: len())
	peptide_sequence_list = dataset[['index', 'text']].to_numpy()
	embed = []
	num_sequences = len(peptide_sequence_list)
	for i in range(0, num_sequences, 8):
		batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list[i: i + 8])
		batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
		batch_tokens = batch_tokens.to(device)
		with torch.no_grad():
			results = model(batch_tokens, repr_layers=[layer], return_contacts=True)
		token_representations = results["representations"][layer].cpu()
		for idx, item in enumerate(token_representations):
			embed.append([item, batch_lens[idx]])
	return embed


def get_cmap(pdb_folder, seq_ids, seqs, threshold, add_self_loop=False, normalize_dist=True):

    print('getting contact map ...')

    if pdb_folder[-1] != '/':
        pdb_folder += '/'

    list_A = []
    list_E = []

    parser = PDBParser(QUIET=True)

    for seq_id, seq in zip(seq_ids, seqs):
        pdb = str(seq_id) + '.pdb'
        pdb_path = pdb_folder + pdb

        structure = parser.get_structure(str(seq_id), pdb_path)

        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue and residue.id[0] == ' ':
                        ca_coords.append(residue['CA'].get_coord())

        coords = np.array(ca_coords, dtype=np.float32)
        seq = str(seq)

      
        L = min(len(coords), len(seq))
        coords = coords[:L]
        seq = seq[:L]

        dist_matrix = spatial.distance.cdist(coords, coords, 'euclidean')

        A = np.zeros((L, L), dtype=np.int64)
        A[dist_matrix < threshold] = 1

        if add_self_loop:
            np.fill_diagonal(A, 1)
        else:
            np.fill_diagonal(A, 0)

    
        E = np.zeros((L, L, 6), dtype=np.float32)

        for i in range(L):
            for j in range(L):
                if A[i, j] == 0:
                    continue

                dist = float(dist_matrix[i, j])

                
                if i == j or dist < 1e-6:
                    direction = np.zeros(3, dtype=np.float32)
                else:
                    direction = (coords[j] - coords[i]) / (dist + 1e-6)
                    direction = direction.astype(np.float32)

            
                if normalize_dist:
                    dist_feature = dist / threshold
                else:
                    dist_feature = dist

           
                seq_adj = 1.0 if abs(i - j) == 1 else 0.0

                rel_seq_pos = (j - i) / max(L - 1, 1)

                E[i, j] = np.array(
                    [
                        direction[0],
                        direction[1],
                        direction[2],
                        dist_feature,
                        seq_adj,
                        rel_seq_pos
                    ],
                    dtype=np.float32
                )

        list_A.append(A)
        list_E.append(E)

    return list_A, list_E

def to_parse_matrix(A, X, E, Y, eps=1e-6):
    """
    :param A: Adjacency matrix with shape (n_nodes, n_nodes)
    :param E: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param X: node embedding with shape (n_nodes, n_node_features)
    :return:
    """
    num_row, num_col = A.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                e_vec.append(E[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(X, dtype=torch.float32)
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32)
	
    y = torch.tensor([Y], dtype=torch.float32)      
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
	

def handcrafted_features(peptide_sequence_list):
	feature_list = []
	for peptide in tqdm(peptide_sequence_list):
		l = len(peptide)
		fasta_str = [['1', peptide]]
		
		# BINARY
		output = BINARY(fasta_str)[1]
		output.remove(output[0])
		Binary_features = np.array(output).reshape(l, -1)

		feature = {
			'local': np.concatenate([Binary_features], axis=1, dtype=np.float32)
		}
		feature_list.append(feature)
	
	return feature_list

def compute_representations(embeddings,list_A, list_E, label_list, num_types, aaindex_features, position_features,feature_list,AAC_features,blosum62_feature_list):
    sequence_representations = []
    max_len = 256
    aaindex_dim = 553
    for i, embedding in enumerate(embeddings):
        x1, tokens_len = embedding

        x, tokens_len = x1[:256], min(256, tokens_len)

        x = np.pad(x, ((0, 256 - len(x)), (0, 0)), 'constant', constant_values=(0))
        # print(x.shape)
        # input()
        A_real = list_A[i]
        E_real = list_E[i]
        
        L = A_real.shape[0]
        aaindex_ecoding_features = aaindex_features[i] #图
        position_encoding_features = position_features[i] #tu
        aac_features = AAC_features[i] #AAC
        blosum62_encoding_features = blosum62_feature_list[i]
       
        local_features1 = feature_list[i]['local'] 
    
        # #aac_features = aac[i] #AAC
        # local_features = local_features1[:256]
        # local_features = np.pad(local_features, (0, 256 - len(local_features)), 'constant', constant_values=(0))
        aaindex_feature = aaindex_ecoding_features[:256]
        
        aaindex_feature = np.pad(aaindex_feature, ((0, 256 - len(aaindex_feature)), (0, 0)), 'constant', constant_values=(0)) #序列

        #print(aaindex_feature.shape)
		
        X_node = x1[1 : L + 1].numpy()
        
        mask = [False] * tokens_len + [True] * (256 - tokens_len)
        mask = np.array(mask)
        y = np.zeros(num_types)
        y[label_list[i]] = 1
        graph_data = to_parse_matrix(A_real, X_node, E_real, y)
        graph_data.embed_features = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        graph_data.aa_feat = torch.tensor(aaindex_ecoding_features, dtype=torch.float32)
        graph_data.pos_feat = torch.tensor(position_encoding_features, dtype=torch.float32)
        graph_data.aac_feat = torch.tensor(aac_features, dtype=torch.float32)
        graph_data.blosum62_feat = torch.tensor(blosum62_encoding_features,dtype=torch.float32)
        graph_data.local_feat = torch.tensor(local_features1, dtype=torch.float32)
        #graph_data.aac_feat = torch.tensor(aac_features, dtype=torch.float32)
        graph_data.seqlen = torch.tensor([tokens_len], dtype=torch.long)
        graph_data.mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

        sequence_representations.append(graph_data)

    return sequence_representations

class MyDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_type,
        pdb_dir,
        pkl_dir=None
    ):
        # 统一读取TSV
        data = pd.read_csv(
            data_dir,
            sep='\t',
            dtype={'index': str}
        )

        data['text'] = (
            data['text']
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if data_type == 'binary':
            data['label'] = data['label'].astype(int)

        else:
            self.label2id = {
                label: idx
                for idx, label in enumerate(all_labels)
            }

            data['label'] = data['label'].apply(
                lambda label_text: [
                    self.label2id[name.strip()]
                    for name in str(label_text).split(',')
                ]
            )


        label_list = data['label'].tolist()
        seq_ids = data['index'].tolist()
        seq_list = data['text'].tolist()

        # 根据PDB构建图
        list_A, list_E = get_cmap(
            pdb_folder=pdb_dir,
            seq_ids=seq_ids,
            seqs=seq_list,
            threshold=10.0,
            add_self_loop=False,
            normalize_dist=True
        )

        # ESM特征
        if pkl_dir is not None:
            print(f'Load pickle from {pkl_dir}')

            with open(pkl_dir, 'rb') as file:
                embeddings = pickle.load(file)
        else:
            print('Compute feature embedding')
            embeddings = esm_embeddings(data)

        num_types = (
            2
            if data_type == 'binary'
            else len(all_labels)
        )

        aaindex_features = aaindex_ecoding(seq_list)
        position_features = position_encoding(seq_list)
        feature_list = handcrafted_features(seq_list)
        AAC_features = AAC_encoding(seq_list)
        blosum62_feature_list = blosum62_encoding(seq_list)

        self.data = compute_representations(
            embeddings,
            list_A,
            list_E,
            label_list,
            num_types,
            aaindex_features,
            position_features,
            feature_list,
            AAC_features,
            blosum62_feature_list
        )

        print(
            f'Dataset: {data_type}, '
            f'Total {len(self)}'
        )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def test():
	# Test binary classification
	train_dataset = MyDataset(
		data_dir = '/root/autodl-tmp/DUO-ACP-main/datasets/ACP-Mixed-80/ACP-Mixed-80-train.tsv', 
		pdb_dir = '/root/autodl-tmp/DUO-ACP-main1/datasets/ACP-Mixed-80/train_pdb_esmfold',
		data_type = 'binary'
	)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
	for batch in train_loader:
		print(len(batch.embed_features), batch.mask.shape, batch.y.shape)
		
	# Test multi-label classification	
	train_dataset = MyDataset(
		'MLC_datasets/10fold/train_1.fasta', 
		data_type='multiclass'
	)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
	for batch in train_loader:
		print(batch.embed_features.shape, batch.mask.shape, batch.y.shape)
		
	

if __name__ == '__main__':
	test()
