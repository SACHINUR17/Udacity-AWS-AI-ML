import torch
from torch.utils.data import Dataset, DataLoader, random_split

class IMDBDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        """
        Initialize the IMDB dataset with tokenized data
        Args:
            input_ids (torch.Tensor): Token IDs from BERT tokenizer
            attention_mask (torch.Tensor): Attention mask from BERT tokenizer
            labels (torch.Tensor): Binary sentiment labels (0 for negative, 1 for positive)
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        
    def __len__(self):
        """
        Get the length of the dataset
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.labels)

def get_data_loaders(batch_size=8, val_split=0.2, seed=42, num_workers=0):
    """
    Load the tokenized data and create data loaders for training and validation
    Args:
        batch_size (int): Batch size for training and validation
        val_split (float): Proportion of data to use for validation (0.0 to 1.0)
        seed (int): Random seed for reproducibility
        num_workers (int): Number of worker processes for data loading
    Returns:
        tuple: (train_loader, val_loader) - DataLoader objects for training and validation
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Load tokenized data
    try:
        input_ids = torch.load('input_ids.pt')
        attention_mask = torch.load('attention_mask.pt')
        labels = torch.load('labels.pt')
        print(f"Data loaded successfully:")
        print(f"- Input IDs shape: {input_ids.shape}")
        print(f"- Attention mask shape: {attention_mask.shape}")
        print(f"- Labels shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
        
    # Create dataset
    dataset = IMDBDataset(input_ids, attention_mask, labels)
    
    # Split into train and validation sets
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )
    print(f"Dataset split into {train_size} training samples and {val_size} validation samples")
    
    # Create DataLoaders
    # The num_workers parameter is now passed from the function arguments
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# If this file is run directly, test the data loaders
if __name__ == "__main__":
    # Test with no multiprocessing
    train_loader, val_loader = get_data_loaders(num_workers=0)
    
    # Test a batch from the training loader
    for batch in train_loader:
        print("Sample batch shapes:")
        print(f"- Input IDs: {batch['input_ids'].shape}")
        print(f"- Attention mask: {batch['attention_mask'].shape}")
        print(f"- Labels: {batch['labels'].shape}")
        break