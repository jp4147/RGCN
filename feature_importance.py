from load_data import DataLoader
from training import Training
import torch

from lime.lime_text import LimeTextExplainer
from interpretation import MyLime
from tqdm import tqdm
import pandas as pd

class Feature_Imp:
    def __init__(self, model_name, hr):
        torch.cuda.set_device(0)
        self.device = torch.device('cuda')
        
        # Load data
        data_type = model_name.split('_')[-1]
        if 'gnn' in model_name:
            data_loader = DataLoader(data_type = data_type, use_graph_embeddings=True)
        else:
            data_loader = DataLoader(data_type = data_type, use_graph_embeddings=False)
            
        data = data_loader.reidx_dat
        pat_c2i = data_loader.pat_c2i
        self.pat_i2c = {v:k for k,v in pat_c2i.items()}
        vocab_size = data_loader.vocab_size
        
        # Load model
        training = Training(data, vocab_size = vocab_size, device = self.device)
        self.model = training.model
        self.model.load_state_dict(torch.load('model/'+model_name+'.pt'))
        
        # Configure dataset for LIME
        self.month2idx = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
        self.pat_data = {}
        for m, info in hr.items():
            self.pat_data[m] = {}
            patids = info['patids']
            for p in patids:
                seq = data[p]['concept']
                age = data[p]['age']
                la_idx = self.month2idx[m]
                self.pat_data[m][p] = (seq, age, la_idx)
                
        # LIME
        self.explainer = LimeTextExplainer(class_names=["Low Risk", "High Risk"])
    
    def generate_feature_scores(self):
        feature_importances = {}
        for m, patD in tqdm(self.pat_data.items()):
            feature_importances[m] = {}
            for patid, seq_age_i in patD.items():
                feature_importances[m][patid] = self.individual_test(seq_age_i, show = 0)
        return feature_importances

    def individual_test(self, pat_data_sample, show = 1):
        seq = self.seq2str(pat_data_sample[0])
        age = pat_data_sample[1]
        la_idx = pat_data_sample[2]
        my_lime = MyLime(seq, self.model, age, self.device, la_idx)
        exp = self.explainer.explain_instance(seq, my_lime.model4lime)
        if show == 1:
            exp.show_in_notebook(text=True)
        else:
            return exp.as_list()
        
    def seq2str(self, seq):
        seq_str = ' '.join(map(str, seq))
        return seq_str
            
    def avg_rank_sum(self, feature_importances):
        rank_sums = {}
        count_features = {}
        try:
            concept = pd.read_csv('graph_data/concept.csv')
            concept_dict = dict(zip(concept['concept_id'], concept['concept_name']))
        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print("Error: The data file does not exist.")
        except Exception as e:
            # Handle other potential exceptions
            print(f"An error occurred: {e}")
        
        for m, feas in feature_importances.items():
            # Sort features based on their scores for each patient
            rank_sums[m] = {}
            count_features[m] = {}
            for patid, feature_scores in feas.items():
                sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)

                # Add rank to rank_sums and count occurrences
                for rank, (feature, importance) in enumerate(sorted_features,1):
                    if feature not in rank_sums[m]:
                        rank_sums[m][feature] = 0
                        count_features[m][feature] = 0
                    rank_sums[m][feature] += rank
                    count_features[m][feature] += 1
                    
        # Calculate average rank for each feature
        sorted_average_ranks = {}
        for m in feature_importances.keys():
            average_ranks = {concept_dict[self.pat_i2c[int(feature)]]: rank_sums[m][feature] / count_features[m][feature] for feature in rank_sums[m]}

            # Sort features by average rank
            sorted_average_ranks[m] = sorted(average_ranks.items(), key=lambda x: x[1])
            
        return sorted_average_ranks
        