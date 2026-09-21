import torch
from torch.utils.data import Dataset, DataLoader

class IMDBTestDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        """
        Initialize the IMDB test dataset
        
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

# Load the test data
try:
    input_ids = torch.load('test_input_ids.pt')
    attention_mask = torch.load('test_attention_mask.pt')
    labels = torch.load('test_labels.pt')
    
    print(f"Test data loaded successfully:")
    print(f"- Test input IDs shape: {input_ids.shape}")
    print(f"- Test attention mask shape: {attention_mask.shape}")
    print(f"- Test labels shape: {labels.shape}")
except Exception as e:
    print(f"Error loading test data: {str(e)}")
    raise

# Create test dataset and data loader
test_dataset = IMDBTestDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,  # You can adjust this batch size as needed
    shuffle=False,  # No need to shuffle test data
    num_workers=2,
    pin_memory=True
)

print(f"Test DataLoader created with {len(test_dataset)} samples")

# If this file is run directly, test the data loader
if __name__ == "__main__":
    # Test a batch from the test loader
    for batch in test_loader:
        print("Sample test batch shapes:")
        print(f"- Input IDs: {batch['input_ids'].shape}")
        print(f"- Attention mask: {batch['attention_mask'].shape}")
        print(f"- Labels: {batch['labels'].shape}")
        break