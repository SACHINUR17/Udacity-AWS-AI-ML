import pandas as pd
import torch
from transformers import AutoTokenizer

# Load your dataset
df = pd.read_csv("imdb_train_backtranslated2.csv")

# Load a pretrained tokenizer (bert-base-uncased)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the reviews
tokenized = tokenizer(
    df['review'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=100,  # consistent length for each review
    return_tensors='pt'
)

# Save tokenized tensors to files
torch.save(tokenized['input_ids'], 'input_ids.pt')
torch.save(tokenized['attention_mask'], 'attention_mask.pt')

# Save labels
labels_tensor = torch.tensor(df['label'].values)
torch.save(labels_tensor, 'labels.pt')

# Print shapes to verify
print("input_ids shape:", tokenized['input_ids'].shape)
print("attention_mask shape:", tokenized['attention_mask'].shape)
print("labels shape:", labels_tensor.shape)