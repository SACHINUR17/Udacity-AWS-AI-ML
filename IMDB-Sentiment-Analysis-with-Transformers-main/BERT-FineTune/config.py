from transformers import AutoTokenizer, AutoModel
from model import BERTBasedClassifier # This is your new model class

# Load the same tokenizer you used for consistency
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define hyperparameters
# Note: For the optimized model, we don't need num_heads and num_layers
NUM_CLASSES = 2
DROPOUT = 0.1

# Instantiate the new BERT-based model with simplified parameters
model = BERTBasedClassifier(
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

# Print model information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')