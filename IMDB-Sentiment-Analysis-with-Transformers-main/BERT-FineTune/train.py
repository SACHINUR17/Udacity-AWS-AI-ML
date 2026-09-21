import torch.optim as optim
import torch
import torch.nn as nn
from config import model  # This should import the optimized model
from dataloader import get_data_loaders
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import os

def main():
    # Get data loaders with reduced workers for macOS
    train_loader, val_loader = get_data_loaders(batch_size=8, num_workers=0)
    
    # Loss function and optimizer with L2 regularization (weight decay)
    criterion = nn.CrossEntropyLoss()
    
    # Device setup
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Use MPS for training if available
    if torch.backends.mps.is_available():
        train_device = torch.device("mps")
        print("Using MPS for training")
    else:
        train_device = torch.device("cpu")
        print("MPS not available, using CPU for training")
    
    # Always use CPU for validation
    val_device = torch.device("cpu")
    print("Using CPU for validation")
    
    # Move model to training device
    model.to(train_device)
    
    # Initialize optimizer after moving model to device
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # Track losses and accuracy
    train_losses = []
    val_losses = []
    val_accuracies = []
    epoch_num = []
    
    # Early Stopping Setup
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    # Training Loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(train_device)
            attention_mask = batch['attention_mask'].to(train_device)
            labels = batch['labels'].to(train_device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epoch_num.append(epoch)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
        
        # Switch model to CPU for validation
        model.to(val_device)
        
        # Validation Step
        val_loss, val_acc = evaluate(model, criterion, val_loader, val_device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Move model back to training device
        model.to(train_device)
        
        # Early Stopping and Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model to CPU for compatibility
            model.to("cpu")
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")
            # Move model back to training device
            model.to(train_device)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Plot Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_num, train_losses, 'o-', label='Training Loss')
    plt.plot(epoch_num, val_losses, 'o-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    # Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_num, val_accuracies, 'o-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.close()

# Evaluation Function
def evaluate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = total_loss / len(val_loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy

if __name__ == "__main__":
    # This ensures that the multiprocessing of DataLoader works correctly on macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()