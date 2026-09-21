import torch
import torch.nn as nn
from config import model
from test_dataloader import test_loader

def load_model(model_path, device):
    """
    Load the trained model from a file
    Args:
        model_path (str): Path to the saved model weights
        device (torch.device): Device to load the model onto
    Returns:
        model: Loaded model ready for evaluation
    """
    # Load state dict on CPU first to avoid MPS compatibility issues
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # Then move to target device
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset
    Args:
        model: The loaded model to evaluate
        dataloader: DataLoader containing the test data
        device (torch.device): Device to run evaluation on
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Track predictions for further analysis
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Try-except block to handle any MPS compatibility issues
            try:
                outputs = model(input_ids, attention_mask)
            except RuntimeError as e:
                if "MPS" in str(e):
                    print("Falling back to CPU for this batch due to MPS error")
                    # Move everything to CPU and try again
                    model_cpu = model.to('cpu')
                    input_ids_cpu = input_ids.to('cpu')
                    attention_mask_cpu = attention_mask.to('cpu')
                    labels_cpu = labels.to('cpu')
                    
                    outputs = model_cpu(input_ids_cpu, attention_mask_cpu)
                    # Move outputs back to original device
                    outputs = outputs.to(device)
                    # Move model back to original device
                    model.to(device)
                else:
                    raise e
                
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels for further analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
    
    # Additional analytics on test results (optional)
    print(f"Total samples evaluated: {total}")
    print(f"Correct predictions: {correct}")
    
    # Return metrics for potential logging
    return {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "predictions": all_predictions,
        "true_labels": all_labels
    }

def main():
    # Best approach: use CPU for testing to avoid MPS compatibility issues
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model_path = "best_model.pth"
    model_loaded = load_model(model_path, device)
    
    # Evaluate and print results
    metrics = evaluate_model(model_loaded, test_loader, device)
    
    # You could save or further analyze the results here
    # For example:
    # import numpy as np
    # from sklearn.metrics import classification_report
    # print(classification_report(metrics["true_labels"], metrics["predictions"]))

if __name__ == "__main__":
    main()