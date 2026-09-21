import torch
from torch.utils.data import Dataset, DataLoader, random_split

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

# Create dataset
dataset = IMDBDataset(input_ids, attention_mask, labels)

# Split into train and validation datasets (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # No need to shuffle validation data



'''import torch
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
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)'''
