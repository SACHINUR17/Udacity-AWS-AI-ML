import torch
from torch.utils.data import Dataset, DataLoader

input_ids = torch.load('input_ids.pt')
attention_mask = torch.load('attention_mask.pt')
labels = torch.load('labels.pt')

class IMDBDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

dataset = IMDBDataset(input_ids, attention_mask, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
