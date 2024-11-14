import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        # Linear layers for query, key, and value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # Output linear layer
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.size()
        B, T_k, _ = key.size()
        # Compute queries, keys, and values
        q = self.q_linear(query).view(B, T_q, self.num_heads, self.d_k).transpose(1,2)  # (B, num_heads, T_q, d_k)
        k = self.k_linear(key).view(B, T_k, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(value).view(B, T_k, self.num_heads, self.d_k).transpose(1,2)
        # Calculate attention scores
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # Shape: (B, num_heads, T_q, T_k)
        if mask is not None:
            # mask shape: (B, T_q, T_k)
            mask = mask.unsqueeze(1)  # (B, 1, T_q, T_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-1)
        # Compute the weighted sum of values
        out = (attn @ v).transpose(1,2).contiguous().view(B, T_q, -1)
        return self.fc_out(out)

class ReLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        # First linear layer expands the dimensionality
        self.fc1 = nn.Linear(d_model, d_ff)
        # Second linear layer projects back to d_model
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply first linear layer and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Project back to original dimension
        return self.fc2(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = ReLUFeedForward(d_model, d_ff, dropout)
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply layer normalization before attention
        norm_x = self.norm1(x)
        # Apply multi-head attention with residual connection
        attn_out = self.attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_out)
        # Apply layer normalization before feed-forward
        norm_x = self.norm2(x)
        # Apply feed-forward network with residual connection
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = ReLUFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Self-attention with masking (prevent future positions)
        norm_x = self.norm1(x)
        self_attn_out = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(self_attn_out)

        # Encoder-decoder attention
        norm_x = self.norm2(x)
        enc_dec_attn_out = self.enc_dec_attn(norm_x, enc_output, enc_output, memory_mask)
        x = x + self.dropout(enc_dec_attn_out)

        # Feed-forward network
        norm_x = self.norm3(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out)

        return x

class UniversalTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, num_heads, d_ff, dropout, max_len):
        super().__init__()
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Positional embeddings
        self.pos_emb = nn.Embedding(max_len, d_model)
        # Single encoder layer to be shared across all layers
        self.layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T = x.size()
        # Generate position indices
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # Combine token and positional embeddings
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        # Apply shared encoder layer repeatedly
        for _ in range(self.n_layers):
            x = self.layer(x, mask)
        return x

class UniversalTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, num_heads, d_ff, dropout, max_len):
        super().__init__()
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Positional embeddings
        self.pos_emb = nn.Embedding(max_len, d_model)
        # Single decoder layer to be shared across all layers
        self.layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        B, T = x.size()
        # Generate position indices
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # Combine token and positional embeddings
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        # Apply shared decoder layer repeatedly
        for _ in range(self.n_layers):
            x = self.layer(x, enc_output, tgt_mask, memory_mask)
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=512):
        super().__init__()
        # Encoder part
        self.encoder = UniversalTransformerEncoder(src_vocab_size, d_model, n_layers, num_heads, d_ff, dropout, max_len)
        # Decoder part
        self.decoder = UniversalTransformerDecoder(tgt_vocab_size, d_model, n_layers, num_heads, d_ff, dropout, max_len)
        # Final linear layer to generate output logits
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Pass source through the encoder
        enc_output = self.encoder(src, src_mask)
        # Pass target and encoder output through the decoder
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        # Generate output logits
        output = self.fc_out(dec_output)
        return output

def generate_subsequent_mask(size):
    # Create subsequent mask to prevent attention to future tokens
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)
    return mask

def main():
    # Define parameters
    src_vocab_size = 10000    # Size of the source vocabulary
    tgt_vocab_size = 10000    # Size of the target vocabulary
    d_model = 512             # Embedding dimension
    n_layers = 6              # Number of encoder and decoder layers
    num_heads = 8             # Number of attention heads
    d_ff = 2048               # Feed-forward network dimension
    dropout = 0.1             # Dropout rate
    max_len = 512             # Maximum sequence length
    batch_size = 32           # Batch size for testing
    src_seq_length = 20       # Source sequence length for testing
    tgt_seq_length = 20       # Target sequence length for testing

    # Initialize the model
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len
    )

    # Move model to device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create dummy input (batch_size x seq_length)
    src_input_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_length)).to(device)
    tgt_input_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length)).to(device)
    tgt_output_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length)).to(device)

    # Create masks
    src_mask = None  # Assuming no padding tokens for simplicity
    # Generate subsequent mask for target sequence
    tgt_mask = generate_subsequent_mask(tgt_seq_length).to(device)
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, tgt_seq_length, tgt_seq_length)

    # Forward pass
    logits = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)

    # Print the output shape
    print(f"Logits shape: {logits.shape}")  # Expected: [batch_size, tgt_seq_length, tgt_vocab_size]

    # (Optional) Create dummy labels and compute loss
    criterion = nn.CrossEntropyLoss()
    # Reshape logits and targets for loss computation
    logits = logits.view(-1, tgt_vocab_size)
    tgt_output_ids = tgt_output_ids.view(-1)
    loss = criterion(logits, tgt_output_ids)
    print(f"Dummy loss: {loss.item()}")

    # (Optional) Backward pass
    loss.backward()
    print("Backward pass completed successfully.")

if __name__ == '__main__':
    main()
