import numpy as np
import torch

class MyLime: 
    def __init__(self, sequence_str_initial, model, age, device, la_idx):
        self.sequence_str_initial = sequence_str_initial
        self.model = model
        self.age = age
        self.device = device
        self.la_idx = la_idx
        
    def logits_to_probs(self, logits):
        return 1 / (1 + np.exp(-logits))
    
    def model4lime(self, text_inputs):
        predictions = []
        self.model.eval()
    
        # Ensure text_inputs is a list, even if it's just one sample
        if not isinstance(text_inputs, list):
            text_inputs = [text_inputs]
    
        for text_input in text_inputs:
            # Convert the string input back to a tensor
            sequence_list = [int(code) for code in text_input.split(' ') if code.strip()]
            seq_pad = len(self.sequence_str_initial.split(' '))-len(sequence_list)
            seq_tensor = torch.tensor([0]*seq_pad+sequence_list)
        
            # Add an extra dimension to match the input shape expected by the model
            seq_tensor = seq_tensor.unsqueeze(0).to(self.device)
            
            # Assuming age is fixed or predetermined for this explanation
            age_tensor = torch.tensor(self.age)
            age_tensor = age_tensor.unsqueeze(0).to(self.device)
    
            # Get model prediction for this input
            with torch.no_grad():
                prediction = self.model(seq_tensor, age_tensor)[0]           
    
            # Detach the prediction and move to CPU if necessary, then convert to NumPy
            prediction_np = prediction.detach().cpu().numpy()
            pred_prob = self.logits_to_probs(prediction_np[self.la_idx])
            predictions.append(pred_prob)
    
            low_high_prediction = np.array([[1 - pred, pred] for pred in predictions])
    
        # Convert the list of predictions to a NumPy array
        return low_high_prediction