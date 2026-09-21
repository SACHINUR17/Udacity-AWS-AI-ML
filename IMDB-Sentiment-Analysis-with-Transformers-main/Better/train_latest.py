import torch.optim as optim
import torch
import torch.nn as nn
from config import model
from dataloader import train_loader, val_loader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Loss function and optimizer with L2 regularization (weight decay)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Track losses and accuracy
train_losses = []
val_losses = []
val_accuracies = []
epoch_num = []

# Early Stopping Setup
best_val_loss = float('inf')
patience = 5
counter = 0

# Evaluation Function
def evaluate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = total_loss / len(val_loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy

# Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    epoch_num.append(epoch)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation Step
    val_loss, val_acc = evaluate(model, criterion, val_loader, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Early Stopping and Model Saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Plot Loss Curves
plt.figure()
plt.plot(epoch_num, train_losses, 'o-', label='Training Loss')
plt.plot(epoch_num, val_losses, 'o-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()
