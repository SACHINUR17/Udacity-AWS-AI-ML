import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_random_directions(model):
    direction1 = []
    direction2 = []
    for param in model.parameters():
        d1 = torch.randn_like(param)
        d2 = torch.randn_like(param)
        # Gram-Schmidt orthogonalization
        d2 = d2 - (torch.sum(d1 * d2) / torch.sum(d1 * d1)) * d1
        direction1.append(d1)
        direction2.append(d2)
    return direction1, direction2

def perturb_model(model, direction, alpha):
    # Create a copy of the model and perturb it
    model_perturbed = copy.deepcopy(model)
    for param, d in zip(model_perturbed.parameters(), direction):
        param.data.add_(alpha * d)
    return model_perturbed

def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Choose the central point (trained model parameters)
central_model = model

# Get two random directions
direction1, direction2 = get_random_directions(central_model)

# Define the range for alpha and beta
alpha_range = np.linspace(-1.0, 1.0, 50)
beta_range = np.linspace(-1.0, 1.0, 50)

# Calculate losses
losses = np.zeros((len(alpha_range), len(beta_range)))
for i, alpha in enumerate(alpha_range):
    for j, beta in enumerate(beta_range):
        # Perturb the model
        perturbed_model = perturb_model(central_model, direction1, alpha)
        perturbed_model = perturb_model(perturbed_model, direction2, beta)
        # Compute the loss
        loss = compute_loss(perturbed_model, train_loader, criterion, device)
        losses[i, j] = loss

# Plot the loss landscape
plt.figure(figsize=(10, 8))
plt.contourf(alpha_range, beta_range, losses, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('2D Loss Landscape')
plt.show()
