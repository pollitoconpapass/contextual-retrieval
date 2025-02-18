import os
from dotenv import load_dotenv
from controllers.embedding_controller import EmbedingController
from controllers.document_processing_controller import TFIDFController, PineconeController


load_dotenv('./.env')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

TFIDF_INDEX_NAME = os.getenv("TFIDF_INDEX_NAME")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

tfidf = TFIDFController(TFIDF_INDEX_NAME)
pinecone_admin = PineconeController(pinecone_api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
embedding_admin = EmbedingController(model_name=EMBEDDING_MODEL)

def extract_from_pinecone(query: str, top_k: int = 5):
    query_embedding = embedding_admin.generate_embeddings(query) 
    results = pinecone_admin.load_and_query_pinecone(query_embedding, top_k)
    return results

def extract_from_tfidf(query: str, top_k: int = 5):
    results = tfidf.load_and_query_tfidf(query, top_k)
    return results

if __name__ == "__main__":
    query = """What ancient civilizations does the document talk about emphasizing their early practices of medicine?"""

    print("\n\nTFIDF")
    results_tfidf = extract_from_tfidf(query)
    print(results_tfidf)

    print("\n\nPinecone (Vectorized DB)")
    results_pinecone = extract_from_pinecone(query)
    print(results_pinecone)
