from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

class DataSplit:
    def __init__(self, data, batch = 16):
        self.data = data
        self.batch = batch
        self.ids_train, self.ids_val, self.ids_test = self.split_ids()
        
        print('create datasets')
        self.train_data = self.create_data_list(self.ids_train)
        self.val_data = self.create_data_list(self.ids_val)
        self.test_data = self.create_data_list(self.ids_test)
        
        print('create trainloaders')
        self.train_loader = DataLoader(self.train_data, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        
    def split_ids(self):
        ids, y = [], []
        for k, v in self.data.items():
            ids.append(k)
            y.append(v['label'])
            
        stratify_labels = [''.join(map(str, label)) for label in y]
        ids_train, ids_test, y_train, y_test = train_test_split(ids, y, test_size=0.2, stratify=stratify_labels, random_state=42)
        stratify_labels_train = [''.join(map(str, label)) for label in y_train]
        ids_train, ids_val, y_train, y_val = train_test_split(ids_train, y_train, test_size=0.25, stratify=stratify_labels_train, random_state=42)  # 0.25 x 0.8 = 0.2
        return ids_train, ids_val, ids_test
        
    def collate_fn(self, batch):
        seq=  pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        age=  pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
        label = torch.tensor([item[2] for item in batch])

        return seq, age, label

    def create_data_list(self, ids_subset):
        data_list = []
        for key in ids_subset:
            seq = torch.tensor(self.data[key]['concept'])
            age = torch.tensor(self.data[key]['age'])
            la = self.data[key]['label']
            data_list.append((seq, age, la))
        return data_list
