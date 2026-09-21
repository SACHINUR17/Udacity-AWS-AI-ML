import torch
from torch.utils.data import DataLoader
from config import model, tokenizer  # clearly importing the instantiated model and tokenizer
from dataloader import train_loader as val_loader  # ensure you have a separate validation loader

# Device clearly set to GPU if available (including Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load your trained model weights clearly
model.load_state_dict(torch.load("trained_tx_contxt_aug_new.pth"))
model.to(device)

# Evaluation loop clearly structured
model.eval()
correct = 0
total = 0

with torch.no_grad():  # Ensure gradients are off
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy*100:.2f}%")
