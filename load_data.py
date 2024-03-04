import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from tqdm import tqdm

class DataLoader:
    def __init__(self, data_type='dx', use_graph_embeddings=False):
        self.data_type = data_type
        self.use_graph_embeddings = use_graph_embeddings
        self.networkx_graph_path = 'graph_data/graph_pat_sym_rf_lab.pickle'
        self.node_embeddings_path = 'graph_data/node_embeddings.npy'
        self.rel_embeddings_path = 'graph_data/rel_embeddings.npy'
        
        self.dx_mod_nocancer_path = 'pat_data/dx_mod_nocancer.pickle'
        self.lab_mod_nocancer_path = 'pat_data/lab_mod_nocancer.pickle'
        self.med_mod_nocancer_path = 'pat_data/med_mod_nocancer.pickle'
        
        self.dx_c2i_path = 'pat_data/c2i_dx.pickle'
        self.lab_c2i_path = 'pat_data/c2i_lab.pickle'
        self.med_c2i_path = 'pat_data/c2i_med.pickle'
        
        self.embedding_dim = 32
        self.output_dim = 5  # 5 time points
        self.vocab_size = None
        
        # Load selected data
        if self.data_type == 'dx':
            self.pat_data, self.c2i, self.i2c = self.load_dx()
        elif self.data_type == 'lab':
            self.pat_data, self.c2i, self.i2c = self.load_lab()
        elif self.data_type == 'med':
            self.pat_data, self.c2i, self.i2c = self.load_med()
        
        self.reidx_dat, self.pat_c2i  = self.data_reindex()
        
        if self.use_graph_embeddings:
            self.nodes, self.nx_G = self.load_graph_data()
            self.final_embeddings = self.generate_final_embeddings()
        else:
            self.final_embeddings = None   
            
    def load_dx(self):
        with open(self.dx_mod_nocancer_path, 'rb') as f:
            dx = pickle.load(f)
        with open(self.dx_c2i_path, 'rb') as f:
            c2i = pickle.load(f)
        i2c = {v: k for k, v in c2i.items()}
        return dx, c2i, i2c

    def load_lab(self):
        with open(self.lab_mod_nocancer_path, 'rb') as f:
            lab = pickle.load(f)
        with open(self.lab_c2i_path, 'rb') as f:
            c2i = pickle.load(f)
        i2c = {v: k for k, v in c2i.items()}
        return lab, c2i, i2c 

    def load_med(self):
        with open(self.med_mod_nocancer_path, 'rb') as f:
            med = pickle.load(f)
        with open(self.med_c2i_path, 'rb') as f:
            c2i = pickle.load(f)
        i2c = {v: k for k, v in c2i.items()}
        return med, c2i, i2c  
    
    def load_graph_data(self):
        with open(self.networkx_graph_path, 'rb') as f:
            nx_G = pickle.load(f)
        nodes = np.array(list(sorted(nx_G.nodes())))
        return nodes, nx_G  
    
    def data_reindex(self):
        patlist = list(self.pat_data.keys())
        seq_names = list(self.pat_data[patlist[0]].keys())
        seq = seq_names[0] #concept_dx, concept_lab, concept_med
        age = seq_names[1]
        label = seq_names[2]
        
        print('assign reindex to concepts')
        pat_seq_idx = [v[seq] for k,v in self.pat_data.items()]
        unique_seq_idx = set(idx for idx_lst in pat_seq_idx for idx in idx_lst)
        unique_seq_concept = [self.i2c[i] for i in  unique_seq_idx]
        # unique_seq_concept = list(set(unique_seq_concept))
        pat_c2i = {c: i+1 for i, c in enumerate(unique_seq_concept)}
        
        print('data reindexing')
        reidx_dat = {}
        for pat in tqdm(patlist):
            reidx_dat[pat] = {}
            reidx_dat[pat]['concept'] = [pat_c2i[self.i2c[i]] for i in self.pat_data[pat][seq]]
            reidx_dat[pat]['age'] = self.pat_data[pat][age]
            reidx_dat[pat]['label'] = self.pat_data[pat][label]
            
        print('vocab_size:', len(list(pat_c2i.keys()))+1)
        self.vocab_size = len(list(pat_c2i.keys()))+1
        
        return reidx_dat, pat_c2i
    
    def generate_final_embeddings(self):
        node_embeddings = np.load(self.node_embeddings_path)  
        pat_dx_in_concepts = list(self.pat_c2i.keys())
        node2idx = {n: i for i, n in enumerate(self.nodes)}
        pat_gnn_emb = [node_embeddings[node2idx[c]] for c in pat_dx_in_concepts]
        pre_trained_embeddings = torch.tensor(np.array(pat_gnn_emb), dtype=torch.float32)
        padding_embedding = torch.zeros(1, self.embedding_dim)
        final_embeddings = torch.cat((padding_embedding, pre_trained_embeddings), dim=0)
        return final_embeddings