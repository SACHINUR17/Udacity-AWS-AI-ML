import torch
import torch.nn as nn
from transformers import AutoModel

class BERTBasedClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        # Use the entire BERT model rather than just the embeddings
        # This eliminates the need for custom transformer layers
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        # Get embedding dimension from BERT (fixed at 768 for bert-base-uncased)
        self.embed_dim = 768
        
        # Simple classification head
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # Get outputs from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply layer norm and activation
        output = self.layer_norm(pooled_output)
        output = self.activation(output)
        output = self.dropout(output)
        
        # Classification layer
        logits = self.fc_out(output)
        
        return logits

    def count_parameters(self):
        """Count number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Create the model
model = BERTBasedClassifier(num_classes=2)

# Print model parameter count
total_params = model.count_parameters()
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {total_params:,}")