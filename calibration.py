from load_data import DataLoader
from training import Training
import torch
from evaluate_performance import Evaluate
import numpy as np

import tensorflow as tf
import tensorflow_lattice as tfl
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle
from matplotlib import pyplot as plt

class Calibrate:
    def __init__(self, model_name, calibrate = 0):
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        num_point = 30
        lr = 0.0001
        
        # Load data and model 
        data_type = model_name.split('_')[-1]
        if 'gnn' in model_name:
            data_loader = DataLoader(data_type = data_type, use_graph_embeddings=True)
        else:
            data_loader = DataLoader(data_type = data_type, use_graph_embeddings=False)
            
        data = data_loader.reidx_dat
        vocab_size = data_loader.vocab_size
        
        # Split data from training
        training = Training(data, vocab_size = vocab_size, device = device)
        
        train_data = training.data_splits.train_data
        val_data = training.data_splits.val_data
        test_data = training.data_splits.test_data
        
        train_ids, val_ids, self.test_ids = training.data_splits.split_ids()
        model = training.model
        
        # Scores from evaluation
        ev_val = Evaluate(val_data, val_ids, model, 'model/'+model_name+'.pt')
        self.ev_test = Evaluate(test_data, self.test_ids, model, 'model/'+model_name+'.pt')
        
        self.scores_val = ev_val.raw_scores()
        self.scores_test = self.ev_test.raw_scores()
        
        self.val_true, self.val_prob, _ = self.probs(self.scores_val)
        self.test_true, self.test_prob, _ = self.probs(self.scores_test)
        
        # Start calibrate or load calibrated model
        if calibrate == 1:
            self.calib_model = self.calibration(self.val_prob, self.val_true,
                                                self.test_prob, self.test_true, num_point, lr)
            self.calib_model.save('calibM') 
        else:
            self.calib_model = tf.keras.models.load_model('model/calibM')
            
        self.cal_test_prob = self.calib_model.predict(self.test_prob)
        self.cal_val_prob = self.calib_model.predict(self.val_prob)
        
        self.reliability_plot(self.test_prob, self.cal_test_prob.flatten(), 
                              self.val_prob, self.cal_val_prob.flatten(), 
                              self.test_true, self.val_true)
        
        # Save results
        cal_results = {}
        cal_results['ids'] = np.array(self.test_ids)
        cal_results['true'] = self.test_true
        cal_results['cal_probs'] = self.cal_test_prob.flatten()

        with open('output/cal_results.pickle', 'wb') as handle:
            pickle.dump(cal_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def logits_to_probs(self, logits):
        return 1 / (1 + np.exp(-logits))

    def probs(self, scores):
        
        months2idx = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
        ids, true_label, pred_prob = [], [], []
        for month, idx in months2idx.items():
            ids.extend(scores[month]['ids'])
            true_label.extend([i[idx] for i in scores[month]['labels']])
            pred_prob.extend([i[idx] for i in scores[month]['raw_scores']])
            
        pred_prob = self.logits_to_probs(np.array(pred_prob))
        
        return np.array(true_label), pred_prob, ids
        
    def calibration(self, val_prob, val_true, test_prob, test_true, n, lr):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # PWLCalibration layer.
        input_shape = (1,)
        inputs = tf.keras.Input(shape=input_shape)
        outputs = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0, 1, num=n),
            dtype=tf.float32,
            output_min=0.0,
            output_max=1.0,
            monotonicity='increasing'
        )(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model.
        LEARNING_RATE = lr
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse')  # Using mean squared error for simplicity

        # Fit the model.
        model.fit(val_prob, val_true, validation_data = (test_prob, test_true), epochs=100, callbacks=[early_stopping])
        return model
        
    def reliability_param(self, probs, true_labels, n_bins=10):
        # Create bins
        bins = np.linspace(0, 1, n_bins+1)
        binids = np.digitize(probs, bins) - 1

        bin_sums = np.bincount(binids, weights=probs, minlength=n_bins)
        bin_true = np.bincount(binids, weights=true_labels, minlength=n_bins)
        bin_total = np.bincount(binids, minlength=n_bins)
        
        bin_means = bin_sums / np.maximum(bin_total, 1)
        bin_true_props = bin_true / np.maximum(bin_total, 1)
        return bin_means, bin_true_props

    def reliability_plot(self, p, p_cal, p_tr, p_cal_tr, y, y_tr):
        
        m, prop = self.reliability_param(p, y, )
        m_cal, prop_cal = self.reliability_param(p_cal, y)
        m_tr, prop_tr = self.reliability_param(p_tr, y_tr, )
        m_cal_tr, prop_cal_tr = self.reliability_param(p_cal_tr, y_tr)
        plt.plot(m, prop, 's-', color = 'gray', label='uncalibrated(test set)')
        plt.plot(m_cal, prop_cal, 's-', color = 'k', label = 'calibrated(test set)')
        plt.plot(m_tr, prop_tr, '.--', color = 'gray', label='uncalibrated(validation set)')
        plt.plot(m_cal_tr, prop_cal_tr, '.--', color = 'k', label = 'calibrated(validation set)')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Reliability Plot')
        plt.legend()
        plt.show()
    