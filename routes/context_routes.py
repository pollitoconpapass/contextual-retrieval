import os, sys
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import APIRouter

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from controllers.embedding_controller import EmbedingController
from controllers.document_processing_controller import PineconeController, TFIDFController
from controllers.rank_fusion_controller import RankFusionController, format_pinecone_results, format_tfidf_results

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

router = APIRouter()

class PineconeResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, str]

class QueryPineconeResponse(BaseModel):
    results: List[PineconeResult]


# === EXTRACT CONTEXT FROM PINECONE INDEX ===
@router.post("/context-pinecone")
async def query_pinecone(data: dict):
    """
    query: str
    pinecone_index_name: str
    top_k: int
    """
    query = data.get("query")
    pinecone_index_name = data.get("pinecone_index_name")
    top_k = data.get("top_k")

    try: 
        embedding_admin = EmbedingController(model_name=EMBEDDING_MODEL)
        pinecone_admin = PineconeController(pinecone_api_key=PINECONE_API_KEY, index_name=pinecone_index_name)

        query_embedding = embedding_admin.generate_embeddings(query) 
        results = pinecone_admin.load_and_query_pinecone(query_embedding, top_k)

        matches = results.get('matches', [])
        serialized_results = [PineconeResult(id=match['id'], score=match['score'], metadata=match['metadata']) for match in matches]

        return QueryPineconeResponse(results=serialized_results)

    except Exception as e:
        return {"error": f"Error when trying to extract context from {pinecone_index_name}: {e}"}
        

# === EXTRACT CONTEXT FROM TF-IDF INDEX ===
@router.post("/context-tfidf")
async def query_tfidf(data: dict):
    """
    query: str
    tfidf_index_name: str
    top_k: int
    """
    query = data.get("query")
    tfidf_index_name = data.get("tfidf_index_name")
    top_k = data.get("top_k")

    try:
        tfidf = TFIDFController(tfidf_index_name)
        results = tfidf.load_and_query_tfidf(query, top_k)

        return {"results": results}
    
    except Exception as e:
        return {"error": f"Error when trying to extract context from {tfidf_index_name}: {e}"}


# === RANK FUSION HYBRID ===
@router.post("/rank-fusion")
async def query_hybrid(data: dict):
    """
    query: str
    pinecone_index_name: str
    tfidf_index_name: str
    top_k: int
    """
    query = data.get("query")
    pinecone_index_name = data.get("pinecone_index_name")
    tfidf_index_name = data.get("tfidf_index_name")
    top_k = data.get("top_k", 5)

    try: 
        # Extracting data from Pinecone
        embedding_admin = EmbedingController(model_name=EMBEDDING_MODEL)
        pinecone_admin = PineconeController(pinecone_api_key=PINECONE_API_KEY, index_name=pinecone_index_name)

        query_embedding = embedding_admin.generate_embeddings(query) 
        pinecone_results = pinecone_admin.load_and_query_pinecone(query_embedding, top_k)
        formatted_pinecone_results = format_pinecone_results(pinecone_results['matches'])

        # Extracting data from TF-IDF Index
        tfidf = TFIDFController(tfidf_index_name)
        tfidf_results = tfidf.load_and_query_tfidf(query, top_k)
        formatted_tfidf_results = format_tfidf_results(tfidf_results)

        # Rank fusion
        rank_fusion = RankFusionController()
        fused_results = rank_fusion.reciprocal_rank_fusion([
            formatted_pinecone_results,
            formatted_tfidf_results
        ])

        return {
            "results": fused_results[:top_k],
            "source_results": {
                "pinecone": formatted_pinecone_results,
                "tfidf": formatted_tfidf_results
            }
        }

    except Exception as e:
        return {"error": f"Error when trying to Rerank: {e}"}