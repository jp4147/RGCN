import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from scipy import interpolate

class Evaluate:
    def __init__(self, test_data, ids, model, trained_model_path):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device))
        
        
        self.test_data = test_data
        self.ids = ids
        self.label2month = {5:'3m', 4:'6m', 3:'12m', 2:'36m', 1:'60m'}
        
        self._raw_scores = None
    
    def data_by_label(self):
        y = [i[2] for i in self.test_data]
        y_sum = np.array([sum(i) for i in y])
        
        data_by_label = {}
        id_by_label = {}
        for label_sum in [0,1,2,3,4,5]:
            idx = np.where(y_sum == label_sum)[0]
            data_by_label[label_sum] = [self.test_data[i] for i in idx]
            id_by_label[label_sum] = [self.ids[i] for i in idx]
        return data_by_label, id_by_label
                    
    def raw_scores(self):
        if self._raw_scores is None:
            data_by_label, id_by_label = self.data_by_label()
            res = {}
            for la_sum in tqdm(self.label2month.keys()):
                res[self.label2month[la_sum]] = {}
                all_outputs, all_labels = [], []
                
                test_loader = DataLoader(data_by_label[la_sum]+data_by_label[0], batch_size=16, shuffle=False, collate_fn=self.collate_fn)
                ids = id_by_label[la_sum]+id_by_label[0]
                with torch.no_grad():
                    for sequences, age, labels in test_loader:
                        sequences, age, labels = sequences.to(self.device), age.to(self.device), labels.float().to(self.device)

                        # Forward pass
                        outputs = self.model(sequences, age)
                        all_outputs.extend(outputs.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())
                        

                res[self.label2month[la_sum]]['ids'] = ids
                res[self.label2month[la_sum]]['raw_scores'] = all_outputs
                res[self.label2month[la_sum]]['labels'] = all_labels
            self._raw_scores = res
        return self._raw_scores
      
    def collate_fn(self, batch):
        seq=  pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        age=  pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
        label = torch.tensor([item[2] for item in batch])
        return seq, age, label

    def roc_pr(self, plot = 'no'):
        res = self._raw_scores if self._raw_scores else self.raw_scores()
        colors = ['k', 'g', 'b', 'orange', 'r']
        tpr_collect, fpr_collect, pre_collect, rec_collect = {}, {}, {}, {}
        pr_thre_collect, roc_thre_collect = {}, {}
        for la_sum, m in self.label2month.items():
            fpr, tpr, roc_thre = roc_curve(np.array(res[m]['labels'])[:,5-la_sum], np.array(res[m]['raw_scores'])[:,5-la_sum])
            pre, rec, pr_thre = precision_recall_curve(np.array(res[m]['labels'])[:,5-la_sum], np.array(res[m]['raw_scores'])[:,5-la_sum])

            tpr_collect[m]=tpr
            fpr_collect[m]=fpr
            pre_collect[m]=pre
            rec_collect[m]=rec
            pr_thre_collect[m] = pr_thre
            roc_thre_collect[m] = roc_thre
        if plot=='yes':
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
            for la_sum, m in self.label2month.items():  
                roc_auc = auc(fpr_collect[m], tpr_collect[m])
                roc_label='AUROC '+m+': {:.2f}'.format(roc_auc)
                ax1.plot(fpr_collect[m], tpr_collect[m], color = colors[5-la_sum], label = roc_label)
                
                pr_auc = auc(rec_collect[m], pre_collect[m])
                pr_label = 'AUPR ' + m + ': {:.2f}'.format(pr_auc)
                ax2.plot(rec_collect[m], pre_collect[m], color=colors[5-la_sum], label=pr_label)
                
            # Set titles and labels
            ax1.set_title('ROC Curve')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.legend(loc="lower right")
            
            ax2.set_title('Precision-Recall Curve')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.legend(loc="lower left")

            # Show plot
            plt.tight_layout()
            plt.show()
        
        return tpr_collect, fpr_collect, pre_collect, rec_collect, roc_thre_collect, pr_thre_collect
    
    def sens_spec(self, specAt = 0.99):
        sen_dict = {}
        spec_dict = {}
        tpr, fpr, pre, rec, _, _ = self.roc_pr(plot = 'yes')
        for m in self.label2month.values():
            v_sen, v_spec = [],[]
            spec = (1-fpr[m])
            sen = tpr[m]

            spec_reversed = spec[::-1]
            sen_reversed = sen[::-1]

            sen_95spec = np.interp(specAt, spec_reversed, sen_reversed)
            f = interpolate.interp1d(sen, spec, kind='linear', fill_value="extrapolate")
            spec_95sen = f(specAt)

            sen_dict[m] = sen_95spec
            spec_dict[m] = spec_95sen
            
        print('sensitivity at '+str(specAt)+' specificity:', sen_dict)
        print('specificity at '+str(specAt)+' sensitivity:', spec_dict)
       