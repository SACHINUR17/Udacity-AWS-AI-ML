import torch
import random
import pandas as pd
from transformers import AutoTokenizer
from config import model
import torch.nn.functional as F

# Clearly load your trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.load_state_dict(torch.load("trained_transformer.pth"))
model.to(device)
model.eval()

# Load your test dataset clearly
df_test = pd.read_csv("imdb_test_dataset.csv")

# Load tokenizer clearly
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Select a random review
random_idx = torch.randint(len(df_test), size=(1,)).item()
review_text = df_test.iloc[random_idx]['review']
true_label = df_test.iloc[random_idx]['label']

# Tokenize clearly
encoded = tokenizer(
    review_text,
    padding='max_length',
    truncation=True,
    max_length=100,
    return_tensors='pt'
).to(device)

# Run inference
with torch.no_grad():
    logits = model(encoded['input_ids'])
    probabilities = F.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

# Clearly display results
label_map = {1: "Positive", 0: "Negative"}

print(f"Review: {review_text}\n")
print(f"Actual Label: {label_map[true_label]}")
print(f"Predicted Label: {label_map[predicted_label]}")
