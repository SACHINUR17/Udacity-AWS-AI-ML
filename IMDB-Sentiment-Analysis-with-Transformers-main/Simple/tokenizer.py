import pandas as pd
import torch
from transformers import AutoTokenizer

# Load your dataset (assuming you've saved your DataFrame previously)
df = pd.read_csv("imdb_train_contextual_augmented_new.csv")

# Load a pretrained tokenizer (e.g., 'bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the reviews clearly
tokenized = tokenizer(
    df['review'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=100,  # consistent length for each review, change if you need to, also update CONFIG
    return_tensors='pt'
)

# Save tokenized tensors explicitly to files 

torch.save(tokenized['input_ids'], 'input_ids.pt')
torch.save(tokenized['attention_mask'], 'attention_mask.pt')

# Also clearly save labels
labels_tensor = torch.tensor(df['label'].values)
torch.save(labels_tensor, 'labels.pt')

# Print shapes to verify everything is correctly saved
print("input_ids shape:", tokenized['input_ids'].shape)
print("attention_mask shape:", tokenized['attention_mask'].shape)
print("labels shape:", labels_tensor.shape)
