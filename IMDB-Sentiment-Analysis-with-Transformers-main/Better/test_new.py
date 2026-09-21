import torch
import torch.nn as nn
from config import model
from test_dataloader import test_loader

def load_model(model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def evaluate_model(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with input_ids and attention_mask
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #model_path = "best_model.pth"
    model_path = "best_checkpoint.pth"

    model_loaded = load_model(model_path, device)
    evaluate_model(model_loaded, test_loader, device)

if __name__ == "__main__":
    main()
