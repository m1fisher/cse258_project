import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import pickle
import utils
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Configurations
BATCH_SIZE = 90
EPOCHS = 1
EMBEDDING_DIM = 56
NUM_HEADS = 8
NUM_LAYERS = 6
# SEQ_LENGTH = 4  # Max sequence length
# NUM_SONGS = 100000  # Total unique songs
LEARNING_RATE = 0.001 #ADAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset Preparation
class PlaylistDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): (playlist_id, song_sequence, next_song)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        playlist_id, song_sequence = self.data[idx]
        song_sequence = torch.tensor(song_sequence, dtype=torch.long)
        # next_song = torch.tensor(next_song, dtype=torch.long)
        input_sequence = song_sequence[:-1]
        target_sequence = song_sequence[1:]
        #song_sequence, next_song
        return input_sequence, target_sequence

# Model Definition
class PlaylistTransformer(nn.Module):
    def __init__(self, num_songs, embedding_dim, num_heads, num_layers, seq_length):
        super(PlaylistTransformer, self).__init__()
        self.embedding = nn.Embedding(num_songs, embedding_dim)
        self.position_embedding = nn.Embedding(seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_dim, num_songs)
        self.seq_length = seq_length

    def forward(self, song_sequence):
        """
        Inputs:
            song_sequence: (batch_size, seq_length)
        Returns:
            predictions: (batch_size, seq_length, num_songs)
        """
        batch_size, seq_len = song_sequence.size()
        embeddings = self.embedding(song_sequence)

        positions = torch.arange(seq_len, device=song_sequence.device).unsqueeze(0).repeat(batch_size, 1)
        embeddings += self.position_embedding(positions)

        # Generate causal mask
        #tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(song_sequence.device)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(song_sequence.device)
        
        transformer_out = self.transformer_encoder(
            embeddings,
            mask=mask
        ) 
        predictions = self.fc_out(transformer_out)
        # XXXX -- see below -- Transformer expects (seq_len, batch_size, embedding_dim)
        # Changed for efficiency to (batch_size, seq_len, embedding_dim)
        # embeddings = embeddings.permute(1, 0, 2)
        # transformer_out = self.transformer(embeddings, embeddings)
        # predictions = self.fc_out(transformer_out)
        return predictions #.permute(1, 0, 2)  # Back to (batch_size, seq_len, num_songs)

# Training Loop
def train_model(model, dataloader, optimizer, scheduler, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (input_sequence, target_sequence) in enumerate(dataloader):
            # input_sequence, next_song = song_sequence.to(DEVICE), next_song.to(DEVICE)
            # optimizer.zero_grad()
            input_sequence = input_sequence.to(DEVICE)  # (batch_size, seq_length)
            target_sequence = target_sequence.to(DEVICE)  # (batch_size, seq_length)
            optimizer.zero_grad()

            outputs = model(input_sequence)
            outputs = outputs.reshape(-1, outputs.size(-1))  # (batch_size * seq_length, num_songs)
            target_sequence = target_sequence.reshape(-1)
            # outputs = model(song_sequence)  # (batch_size, seq_length, num_songs)
            # outputs = outputs[:, -1, :]  # Only predict for the last song in the sequence
            loss = criterion(outputs, target_sequence)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            
            writer.add_scalar('Loss/train', loss.item(), (num_epochs+1) * batch_idx)
            writer.add_scalar('Learning rate', current_lr,(num_epochs+1) * batch_idx)
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

def _load_model():
    with open(os.path.join('train_data', "transformer_data.pkl"), "rb") as fh:
        ease = pickle.load(fh)
    return ease

def _load_mapping():
    with open(os.path.join('train_data', "song_to_index.pkl"), "rb") as fh:
        ease = pickle.load(fh)
    return ease

# Inference: Predict Next 500 Songs for a Playlist
def predict_playlist(playlists):
    #playlists
    #{pid: []trackIds}
    model = _load_model()
    song_to_index = _load_mapping()
    index_to_song = {v:k for k,v in song_to_index}
    max_seq_length = len(list(song_to_index))
    model.eval()
    preds = {}
    for pid, seed_tracks in playlists.items(): 
        predictions = [song_to_index[t] for t in seed_tracks]
        for _ in range(500):
            input_sequence = torch.tensor(predictions[-max_seq_length:], dtype=torch.long, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_sequence)  # (1, seq_length, num_songs)
                next_song_logits = outputs[:, -1, :]  # Get logits for the last position
                next_song = torch.argmax(next_song_logits, dim=-1).item()
                #something about mapping here
                predictions.append(index_to_song[next_song])
        preds[pid] = predictions[len(seed_tracks):]
    
    return preds
    # for _ in range(num_predictions - len(seed_songs)):
    #     with torch.no_grad():
    #         outputs = model(seed_tensor)
    #         next_song_logits = outputs[:, -1, :]  # Get logits for the last position
    #         next_song = torch.argmax(next_song_logits, dim=1).item()
    #         predictions.append(next_song)

    #         # Update seed tensor with the new song
    #         if len(predictions) > seq_length:
    #             seed_tensor = torch.tensor(predictions[-seq_length:], dtype=torch.long, device=DEVICE).unsqueeze(0)
    #         else:
    #             seed_tensor = torch.tensor(predictions, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # return predictions

# Simulated Data Generation (Replace with actual data)
def simulate_data():
    data = []
    slices = [x for x in os.listdir('train_data') if x.startswith("mpd.slice")]
    batch_count = 0
    #added slice 1 so it runs, remove for testing
    songsPerPlaylist = defaultdict(list)
    songs = set()
    for filename in slices[:len(slices) //5]: #len(slices) //3
        batch_count += 1
        if (batch_count % 10) == 0:
            print(f"processing batch {batch_count}")
        tracks = utils.read_track_csv(os.path.join('train_data', filename))
        for track in tracks:
            songs.add(track.track_id)
            songsPerPlaylist[track.pid].append(track.track_id)
    
    songId_to_idx = {song_id: idx for idx, song_id in enumerate(sorted(songs))}
    sliding_window = 9 #10 tracks at 0 index
    for playlist, tracks in songsPerPlaylist.items():
        if len(tracks) < sliding_window:
            continue 
        mapped_tracks = [songId_to_idx[track_id] for track_id in tracks]
        for i in range(len(mapped_tracks)):
            if (i+sliding_window == len(mapped_tracks)):
                break
            if (i > 3):
                #limit the amount of data to three sequences per playlist
                break
            #sliding window to capture +1 next sequences, use mask decoder architecture
            #ensure size of slice is always the same and incremnting by 1
            sequence = mapped_tracks[i:i+sliding_window]
            # print(f'playlist {playlist}, sequence {sequence} at {i}')
            data.append((playlist, sequence))
    
    #num songs and song_id_idx should be same length
    num_songs = len(songs)
    song_sequence_num = len(data[0][1])
    print('assert unique songs', len(songs), len(list(songId_to_idx)))
    for _,song_sequence in data:
        assert len(song_sequence) == song_sequence_num
        assert max(song_sequence) < num_songs

    print('entries', len(data))
    return data, num_songs, song_sequence_num, songId_to_idx

# Main Execution
if __name__ == "__main__":
    # Simulate data
    data, NUM_SONGS, SEQ_LENGTH, songId_to_idx = simulate_data()

    print('length sequence', SEQ_LENGTH)
    print('unique songs', NUM_SONGS)

    # Data preparation
    dataset = PlaylistDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model initialization
    #advising seq length -1? why
    model = PlaylistTransformer(NUM_SONGS, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LENGTH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()
    try:
        train_model(model, dataloader, optimizer, scheduler, criterion, num_epochs=EPOCHS)

    except KeyboardInterrupt:
        print("Saving model before exiting...")
        # torch.save(model.state_dict(), "model_interrupt.pth")
        # pickle.dump(model, open(os.path.join('train_data', "transformer_data.pkl"), "wb"))
        # pickle.dump(songId_to_idx, open(os.path.join('train_data', "song_to_index.pkl"), "wb"))
    # Train model
    print('training complete')
    writer.close()
    # save model for a new playlist
    torch.save(model.state_dict(), "model_interrupt.pth")
    pickle.dump(model, open(os.path.join('train_data', "transformer_data.pkl"), "wb"))
    pickle.dump(songId_to_idx, open(os.path.join('train_data', "song_to_index.pkl"), "wb"))
    
    
    # seed_songs = [songId_to_idx[song_id] for song_id in seed_song_ids]  # Provide actual seed song IDs
    # predictions = predict_playlist(model, seed_songs, SEQ_LENGTH - 1, num_predictions=500)
    # predicted_song_ids = [list(songId_to_idx.keys())[list(songId_to_idx.values()).index(idx)] for idx in predictions]
    # print("Predicted Songs:", predicted_song_ids)



    # Predict for a new playlist
    # model = PlaylistTransformer(NUM_SONGS, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LENGTH).to(DEVICE)
    # model.load_state_dict(torch.load('path/to/your/model.pth'))
    # seed_songs = [1, 5, 8, 12, 20]  # Example seed songs
    # predictions = predict_playlist(model, seed_songs, SEQ_LENGTH, num_predictions=500)
    # print("Predicted Songs:", predictions)


