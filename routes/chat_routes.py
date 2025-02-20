import sys, os
from dotenv import load_dotenv
from fastapi import APIRouter

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from controllers.llm_controller import LLMController
from controllers.embedding_controller import EmbedingController
from controllers.custom_rerank_controller import CustomRerankController
from controllers.document_processing_controller import PineconeController, TFIDFController
from controllers.rank_fusion_controller import RankFusionController, format_pinecone_results, format_tfidf_results

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL_NAME")

router = APIRouter()

@router.post("/chat")
async def chat(data: dict):
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

        # Reranking
        reranker = CustomRerankController(model_name='flashrank', language_code='en')
        reranked_results = reranker.rerank(query, fused_results[:top_k])

        # Calling the LLM Assistant
        llm_admin = LLMController(model_name=LLM_MODEL)
        formatted_reranked_results = "\n".join(reranked_results)
        assistant_response = llm_admin.chat_llm(context=formatted_reranked_results, message=query)

        return {"answer": assistant_response}

    except Exception as e:
        print(f"Error when trying to make the chat: {e}")
        return {"error": f"Error when trying to make the chat: {e}"}
