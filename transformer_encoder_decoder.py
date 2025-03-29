import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder class
def positional_encoding(position, d_model):
  pass

def create_encoder():
    class Encoder(nn.Module):
        def __init__(self, vocab_size, input_dim, hidden_dim, num_layers=4, num_heads=8, eps=1e-8):
            super(Encoder, self).__init__()
            self.embed_dim = input_dim
            self.embedding = nn.Embedding(vocab_size, input_dim)
            self.positional_encoding = positional_encoding # it is a method
            self.mha = nn.MultiHeadAttention(input_dim, num_heads)
            self.layer_norm1 = nn.LayerNorm(hidden_dim, eps)
            self.layer_norm2 = nn.LayerNorm(hidden_dim, eps)
            self.activation = nn.ReLU()
            self.FNN = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*4),
                self.activation,
                nn.Linear(hidden_dim*4, hidden_dim)
            )

            self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": self.mha,
                "layer_norm1": self.layer_norm1,
                "fnn": self.FNN,
                "layer_norm2": self.layer_norm2
            }) for _ in range(num_layers)
        ])

        def forward(self, x):
          x = self.embedding(x)
          x = x + self.positional_encoding(x, self.embed_dim)
          # Pass through multiple encoder layers
          for layer in self.layers:
            residual = x
            x = layer["mha"](x, x, x)[0] + residual
            x = layer["layer_norm1"](x)
            residual = x
            x = layer["FNN"](x) + residual
            x = layer["layer_norm2"](x)

          return x

    
    return Encoder

# Define the Decoder class
def create_decoder():
    class Decoder(nn.Module):
        def __init__(self, vocab_size, hidden_dim, num_layers=4, num_heads=8, eps=1e-8):
            super(Decoder, self).__init__()
            self.embed_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.positional_encoding = positional_encoding # it is a method
            self.mha = nn.MultiHeadAttention(hidden_dim, num_heads)
            self.mha_cross = nn.MultiHeadAttention(hidden_dim, num_heads)
            self.layer_norm1 = nn.LayerNorm(hidden_dim, eps)
            self.layer_norm2 = nn.LayerNorm(hidden_dim, eps)
            self.layer_norm3 = nn.LayerNorm(hidden_dim, eps)
            self.activation = nn.ReLU()
            self.FNN = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*4),
                self.activation,
                nn.Linear(hidden_dim*4, hidden_dim)
            )
          
            self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha_self": self.mha,
                "mha_cross": self.mha_cross,
                "layer_norm1": self.layer_norm1,
                "fnn": self.FNN,
                "layer_norm2": self.layer_norm2,
                "layer_norm3": self.layer_norm3
            }) for _ in range(num_layers)
        ])
            self.linear = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, encoder_output, attention_mask = None):
          x = self.embedding(x)
          x = x + self.positional_encoding(x, self.embed_dim)
          # Pass through multiple encoder layers
          for layer in self.layers:
            residual = x
            x = layer["mha"](x, x, x, attention_mask)[0] + residual
            x = layer["layer_norm1"](x)
            residual = x
            x = layer["mha_cross"](x, encoder_output, encoder_output)[0] + residual
            x = layer["layer_norm2"](x)
            x = layer["FNN"](x) + residual
            x = layer["layer_norm3"](x)
          x = self.linear(x)
          x = torch.softmax(x)
          return x
    
    return Decoder
