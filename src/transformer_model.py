import os
import sys

import utils

def create_sequences(data_dir):
    slices = [x for x in os.listdir(data_dir) if x.startswith("mpd.slice")]
    sequences = {}
    batch_count = 0
    for filename in slices[:10]:
        batch_count += 1
        if (batch_count % 10) == 0:
            print(f"processing batch {batch_count}")
        tracks = utils.read_track_csv(os.path.join(data_dir, filename))
        for track in tracks:
            if track.pid not in sequences:
                sequences[track.pid] = [track.track_id]
            else:
                sequences[track.pid].append(track.track_id)
    return list(sequences.values())

def make_token_pairs(sequences):
    input_length = 5  # can be tuned
    output_length = 1
    X = []
    Y = []
    for seq in sequences:
        x = [seq[i:i + input_length] for i in range(0, len(seq), input_length)]
        y = [[seq[i]] for i in range(input_length, len(seq), input_length)]
        # throw out extra x value
        x = x[:-1]
        assert len(x) == len(y)
        X.extend(x)
        Y.extend(y)

# c/o gpt4 for initial pytorch code below

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Example Dataset class
class SequenceDataset(Dataset):
    def __init__(self, sequences, window_size):
        self.inputs = []
        self.targets = []
        for seq in sequences:
            for i in range(len(seq) - window_size):
                self.inputs.append(seq[i:i + window_size])  # Input window
                self.targets.append(seq[i + window_size])   # Next token (output)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

# Hyperparameters
window_size = 5  # Length of input sequence
embedding_dim = 8  # Embedding size
num_heads = 4  # Number of attention heads
num_layers = 2  # Number of transformer layers
batch_size = 32
num_epochs = 10
learning_rate = 0.001



# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        embedded = self.embedding(src)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embedding_dim)
        transformer_output = self.transformer(embedded, embedded)  # (seq_len, batch_size, embedding_dim)
        output = self.fc_out(transformer_output)  # (seq_len, batch_size, vocab_size)
        return output.permute(1, 0, 2)  # Back to (batch_size, seq_len, vocab_size)




if __name__ == "__main__":
    data_dir = sys.argv[1]
    seqs = create_sequences(data_dir)
    make_token_pairs(seqs)

    # Dataset and DataLoader
    dataset = SequenceDataset(seqs, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = 1483760  # Total number of unique tokens
    # Initialize the model, loss function, and optimizer
    model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(len(dataloader))
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(batch_idx)
            if batch_idx == 3:
                import pdb; pdb.set_trace()

            # Move inputs and targets to the appropriate device (e.g., GPU if available)
            inputs, targets = inputs, targets

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)  # Outputs shape: (batch_size, seq_len, vocab_size)

            # Use only the final token of each sequence for next-token prediction
            outputs = outputs[:, -1, :]  # (batch_size, vocab_size)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
