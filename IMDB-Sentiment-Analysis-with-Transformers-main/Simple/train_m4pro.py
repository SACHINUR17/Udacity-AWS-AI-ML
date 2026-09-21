import torch.optim as optim
import torch
import torch.nn as nn
from config import model
from dataloader import train_loader
import matplotlib.pyplot as plt
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
avg_losses = []  # Renamed to avg_losses to avoid confusion
epoch_num = []

# Training loop
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    # Wrap train_loader with tqdm
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
        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    avg_losses.append(avg_loss)  # Add the average loss to our list
    epoch_num.append(epoch)
    
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

# Save your trained model explicitly after training completes
torch.save(model.state_dict(), "trained_tx_contxt_aug_new.pth")
print("Model saved!")

# Plot the loss vs epoch
plt.plot(epoch_num, avg_losses, 'o')  # 'o' specifies circle markers
plt.xlabel('Epoch')
plt.ylabel('Avg Loss')
plt.title('Training Loss Curve')
plt.show()


# We track average loss instead of total loss because it:
# - Makes results comparable regardless of dataset size
# - Provides more interpretable metrics (loss per sample)
# - Enables consistent comparisons between different experiments
# - Aligns with standard practice in machine learning literature
