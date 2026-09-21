from transformers import AutoTokenizer
from model import SimpleTransformerClassifier

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define hyperparameters
VOCAB_SIZE = tokenizer.vocab_size
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
NUM_CLASSES = 2
MAX_SEQ_LENGTH = 100
DROPOUT = 0.1

# Instantiate the Transformer model with dropout
model = SimpleTransformerClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT
)

# Print model information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
