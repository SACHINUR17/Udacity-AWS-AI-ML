import torch
from config import model
from test_dataloader import test_loader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model.load_state_dict(torch.load("trained_tx_contxt_aug_new.pth"))
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
