import pickle
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sklearn.feature_extraction.text import TfidfVectorizer

class PineconeController:
    def __init__(self, pinecone_api_key, index_name):
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Initialize Pinecone index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # Assuming using OpenAI embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(index_name)


    def store_embeddings(self, embeddings, chunks, chunk_metadata=None):
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            vector_data = {
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {
                    'text': chunk,
                }
            }
            
            # Add additional metadata if provided
            if chunk_metadata and i < len(chunk_metadata):
                vector_data['metadata'].update(chunk_metadata[i])
                
            vectors_to_upsert.append(vector_data)
        
        # Upsert vectors in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def start_ingestion_process_pinecone(self, chunks, embeddings, chunk_metadata=None):
        # Store embeddings in Pinecone
        self.store_embeddings(embeddings, chunks, chunk_metadata)

    def load_and_query_pinecone(self, query_embedding: list, top_k: int = 5): 
        # -> you have to pass the embedding of the query as parameter.
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return results


class TFIDFController:
    def __init__(self, index_name):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.index_name = index_name

    def create_tfidf_index(self, chunks):
        # Fit and transform the chunks to create TF-IDF vectors
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(chunks)
        return tfidf_vectors
    
    def store_tfidf_index(self, tfidf_vectors, chunks):
        # Save both the vectorizer and the vectors
        with open(self.index_name, 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'vectors': tfidf_vectors,
                'chunks': chunks
            }, f)
            
    def load_tfidf_index(self):
        with open(self.index_name, 'rb') as f:
            return pickle.load(f)
        
    def start_ingestion_process_tfidf(self, chunks):
        # Create and store TF-IDF index
        tfidf_vectors = self.create_tfidf_index(chunks)
        self.store_tfidf_index(tfidf_vectors, chunks)
        
    def load_and_query_tfidf(self, query: str, top_k: int = 5):
        tfidf_data = self.load_tfidf_index()
        vectorizer = tfidf_data['vectorizer']
        tfidf_vectors = tfidf_data['vectors']
        chunks = tfidf_data['chunks']

        query_vector = vectorizer.transform([query])
        similarities = (query_vector * tfidf_vectors.T).toarray()[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        results = [
            {
                'id': f'chunk_{idx}',
                'score': float(score),
                'text': chunks[idx]
            }
            for idx, score in zip(top_indices, top_scores)
        ]

        return results
    