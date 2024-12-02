#relies on scipy.sparse.coo_matrix
#run python src/sparse_repr.py train_data to generate the sparse matrix
# from functools import lru_cache
import faiss
from cornac.data import Dataset
import cornac
import numpy as np
import os
import sys
import pickle
import utils

def genDataset(data_dir):
    slices = [x for x in os.listdir(data_dir) if x.startswith("mpd.slice")]
    entries = []
    batch_count = 0
    #added slice 1 so it runs, remove for testing
    for filename in slices:
        batch_count += 1
        if (batch_count % 10) == 0:
            print(f"processing batch {batch_count}")
        tracks = utils.read_track_csv(os.path.join(data_dir, filename))
        for track in tracks:
            entries.append((track.pid, track.track_id, 1))
    dataset = Dataset.from_uir(entries)
    print('len entries', len(entries))
    print('unique playlists', len(dataset.user_ids))
    print('unique tracks', len(dataset.item_ids))
    pickle.dump(dataset, open(os.path.join(data_dir, "autoencoder_data.pkl"), "wb"))
    print('dataset complete')

def _load_data(data_dir):
    with open(os.path.join(data_dir, "autoencoder_data.pkl"), "rb") as fh:
        ease = pickle.load(fh)
    return ease

#running into scalability issue
#application memory error in training
#updated batching size to 512
#consider using artist id and album id in the data
def trainBIVAE(data_dir):
    dataset = _load_data(data_dir)
    # print('len entries', len(entries))
    print('unique playlists', len(dataset.user_ids))
    print('unique tracks', len(dataset.item_ids))
   
    bivae = cornac.models.BiVAECF(
        k=10,                # Dimensionality of the latent space
        n_epochs=50,         # Number of training epochs
        batch_size=1024,      # Batch size for training
        learning_rate=0.001, # Learning rate for optimization
        verbose=True         # Print training progress
    )
    bivae.fit(dataset)

    #operating assumption - the order of dataset.item_ids is same as the models item vector output. 
    track_to_embedding = {t: v for t,v in zip(dataset.item_ids,bivae.get_item_vectors())}
    pickle.dump(track_to_embedding, open(os.path.join(data_dir, "bivae_model.pkl"), "wb"))
    print('training complete')

def _load_model(data_dir):
    with open(os.path.join(data_dir, "bivae_model.pkl"), "rb") as fh:
        ease = pickle.load(fh)
    return ease

def predict_with_seed_songs(playlists: dict[list]):
    """
    Predict songs for an unseen playlist using its seed songs.
      
    - Using seed vector averaging in the y
    - TODO Using round robin to better capture diversity

    Returns:
        List of recommended track IDs.
    """
    track_to_embedding = _load_model('train_data')
    print('known tracks',len(track_to_embedding))

    embeddings = np.array(list(track_to_embedding.values())).astype('float32')

    index_to_track = {i: t for i, t in enumerate(track_to_embedding.keys())}
    track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")

    # to make ANN search faster
    # move it to a train phase and use IVF
    # https://www.pinecone.io/learn/series/faiss/faiss-tutorial/
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)

    
    #test
    # print(embeddings[0])
    # print(list(index_to_track.items())[0])
    # track, embed = list(track_to_embedding.items())[0]
    # embedded = embed.reshape(1, -1)
  
    # # print(index, embed)
    # distances, indices = index.search(embedded, 3)
    # tracksrec = [index_to_track[i] for i in indices[0] if index_to_track[i] != track]
    # print('seed track', track, 'embed query', embed)
    # print('tracks rec', tracksrec)

    preds = {}
    for pid, seed_tracks in playlists.items():
        if len(seed_tracks) == 0:
            print(f'playlist {pid}: no seed tracks!')
            preds[pid] = []
            continue
        
        embeddings = []
        exclude = []
        for t in seed_tracks:
            if t.track_id in track_to_embedding:
                embeddings.append(track_to_embedding[t.track_id])
                exclude.append(t.track_id)
        
        if len(embeddings) == 0:
            print(f'playlist {pid}: no embeddings found!')
            preds[pid] = []
            continue
        
        avgEmbedding = np.mean(embeddings, axis=0).reshape(1, -1)
        k = 500 

        _, indices = index.search(avgEmbedding, k+len(exclude))
        recommended_songs = [index_to_track[i] for i in indices[0] if index_to_track[i] not in exclude]
        
        preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                  artist_id=track_to_artist[int(x)], album_id=None) for x in recommended_songs[:500]]

    return preds

if __name__ == "__main__":
    if sys.argv[1] == '--datagen':
        genDataset(sys.argv[2])
    elif sys.argv[1] == '--train':
        trainBIVAE(sys.argv[2])