import pandas as pd
import torch
from transformers import AutoTokenizer

# Load IMDb test dataset
df_test = pd.read_csv("imdb_test_dataset.csv")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

tokenized_test = tokenizer(
    df_test['review'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=100,
    return_tensors='pt'
)

# Save clearly
torch.save(tokenized_test['input_ids'], 'test_input_ids.pt')
torch.save(tokenized_test['attention_mask'], 'test_attention_mask.pt')
torch.save(torch.tensor(df_test['label'].values), 'test_labels.pt')
