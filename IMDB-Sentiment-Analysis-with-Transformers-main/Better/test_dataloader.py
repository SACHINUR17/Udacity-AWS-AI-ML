import torch
from torch.utils.data import Dataset, DataLoader

input_ids = torch.load('test_input_ids.pt')
attention_mask = torch.load('test_attention_mask.pt')
labels = torch.load('test_labels.pt')

class IMDBTestDataset(Dataset):
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

test_dataset = IMDBTestDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(test_dataset, batch_size=8)
