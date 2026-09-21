from transformers import AutoTokenizer
from model import SimpleTransformerClassifier

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define hyperparameters clearly
VOCAB_SIZE = tokenizer.vocab_size
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
NUM_CLASSES = 2  # positive and negative
MAX_SEQ_LENGTH = 512

# Instantiate your Transformer
model = SimpleTransformerClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES, MAX_SEQ_LENGTH)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')