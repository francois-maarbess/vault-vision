import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VaultBrain:
    def __init__(self):
        # We use MiniLM because it runs instantly on CPU without needing massive GPUs
        print("🤖 Initializing Sentence Embedder...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 384 is the exact dimension size output by the MiniLM model
        self.index = faiss.IndexFlatL2(384)
        
        # The ledger that keeps track of the human-readable text and timestamps
        self.metadata = [] 

    def build_multimodal_index(self, multimodal_data):
        """
        Takes the God-Tier data object from the video processor and 
        builds a unified FAISS index for instant search.
        """
        print("🧠 Building Multimodal Vector Brain...")
        texts_to_embed = []
        
        # --- 1. Indexing the Spoken Words ---
        for audio in multimodal_data.get("audio_data", []):
            # We tag it so the AI knows this was spoken, not seen
            text = f"[SPOKEN WORD]: {audio['text']}"
            texts_to_embed.append(text)
            self.metadata.append({
                "type": "audio",
                "start": audio['start'],
                "end": audio['end'],
                "content": audio['text']
            })
            
        # --- 2. Indexing the Visual Descriptions ---
        for vision in multimodal_data.get("visual_data", []):
            # We tag it so the AI knows this was on the screen
            text = f"[ON SCREEN]: {vision['description']}"
            texts_to_embed.append(text)
            self.metadata.append({
                "type": "visual",
                "start": vision['timestamp'],
                "end": vision['timestamp'] + 1, # visual is a single moment
                "content": vision['description'],
                "frame_path": vision.get('frame_path', '')
            })
            
        print(f"🔢 Encoding {len(texts_to_embed)} multimodal data points into vectors...")
        embeddings = self.embedder.encode(texts_to_embed)
        
        print("💾 Saving to FAISS Vector Database...")
        # FAISS requires float32 numpy arrays
        self.index.add(np.array(embeddings).astype('float32'))
        print("✅ Brain is fully mapped and online.")

    def search(self, query, top_k=3):
        """
        The magic search function. Takes a user's typed question, turns it into math, 
        and finds the closest matching video/audio segments.
        """
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for idx in indices:
            if idx != -1: # -1 means no result found
                results.append(self.metadata[idx])
                
        # Sort results chronologically by timestamp so they make sense to the user
        results.sort(key=lambda x: x['start'])
        return results