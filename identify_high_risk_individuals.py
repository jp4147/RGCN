import pickle
from sklearn.metrics import precision_recall_curve
import numpy as np
import tensorflow as tf

class Identify_HR:
    def __init__(self):
        with open('output/cal_results.pickle', 'rb') as handle:
            cal_results = pickle.load(handle)
        with open('output/raw_scores_test.pickle', 'rb') as handle:
            raw_scores_test = pickle.load(handle)
            
        self.month2idx = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
        self.calibM = tf.keras.models.load_model('model/calibM')
            
        test_true = cal_results['true']
        cal_test_prob = cal_results['cal_probs']
        precision, recall, thresholds = precision_recall_curve(test_true, cal_test_prob)
        optimal_threshold, max_f1_score = self.find_optimal_threshold(precision, recall, thresholds)
    
        print(f"Optimal Threshold: {optimal_threshold}")
        print(f"Maximum F1 Score: {max_f1_score}")
        
        self.hr = self.hr_pat(raw_scores_test, max_f1_score)
        with open('output/hr_pat.pickle', 'wb') as handle:
            pickle.dump(self.hr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def find_optimal_threshold(self, precision, recall, thresholds):
        # Example: Choose the threshold that maximizes the F1 score
        f1_scores = 2*(precision*recall) / (precision+recall)
        max_f1_index = np.argmax(f1_scores[:-1])  # exclude last value which corresponds to recall=0
        optimal_threshold = thresholds[max_f1_index]
        return optimal_threshold, f1_scores[max_f1_index]
    
    def logits_to_probs(self, logits):
        return 1 / (1 + np.exp(-logits))
    
    def hr_pat(self, raw_scores, f1):
        hr_pat = {}
        for month, data in raw_scores.items():
            hr_pat[month] = {}
            scores = [i[self.month2idx[month]] for i in data['raw_scores']]
            scores = [self.logits_to_probs(i) for i in scores]
            scores_cal = self.calibM.predict(scores)
            scores_cal = scores_cal.flatten()
            indices = np.where(scores_cal > f1)[0]
            
            hr_patids = np.array(data['ids'])[indices]
            hr_scores = scores_cal[indices]
            hr_label = [i[self.month2idx[month]] for i in np.array(data['labels'])[indices]]
            
            hr_pat[month]['patids'] = hr_patids
            hr_pat[month]['scores'] = hr_scores
            hr_pat[month]['labels'] = hr_label
        return hr_pat