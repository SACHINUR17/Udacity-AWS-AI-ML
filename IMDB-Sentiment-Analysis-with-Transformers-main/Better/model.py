import torch
import torch.nn as nn
import math

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_length=100, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)  # LayerNorm for stabilization
        self.activation = nn.GELU()  # GELU is smoother than ReLU, commonly used in transformers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(self.embed_dim)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)

        # Transformer and Layer Norm
        transformer_input = embedded.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(transformer_input)
        output = transformer_output[0]
        
        # Apply Layer Norm and Activation
        output = self.layer_norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        logits = self.fc_out(output)
        return logits

# Positional Encoding remains unchanged
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
