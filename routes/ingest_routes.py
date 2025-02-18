import os, sys
from dotenv import load_dotenv
from fastapi import APIRouter

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from controllers.llm_controller import LLMController
from controllers.embedding_controller import EmbedingController
from controllers.document_reading_controller import DocumentExtractionController
from controllers.document_processing_controller import PineconeController, TFIDFController

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL_NAME")

router = APIRouter()

# === INGESTION PROCESS ===
@router.post("/ingestion")
async def ingestion(data: dict): 
    """
    pdf_path: str
    pinecone_index_name: str
    tfidf_index_name: str
    """
    pdf_path = data.get("pdf_path")
    start_page = data.get("start_page", 0)
    end_page = data.get("end_page", None)
    pinecone_index_name = data.get("pinecone_index_name")
    tfidf_index_name = data.get("tfidf_index_name")

    try: 
        # === DECLARE ALL THE OBJECTS ===
        reading = DocumentExtractionController(pdf_path)
        embedding_admin = EmbedingController(model_name=EMBEDDING_MODEL)
        llm_admin = LLMController(model_name=LLM_MODEL)
        pinecone_admin = PineconeController(pinecone_api_key=PINECONE_API_KEY, index_name=pinecone_index_name)
        tfidf_admin = TFIDFController(tfidf_index_name)


        # Extracting Text from the PDF doc
        text_extracted = reading.extract_text_from_pdf(start_page=start_page, end_page=end_page)

        # Generate chunks
        original_chunks = reading.generate_chunks(text=text_extracted)
        print(f"Number of original chunks: {len(original_chunks)}")

        # Generate main idea
        text_main_idea = llm_admin.generate_main_text_idea(text=text_extracted)

        # Give context to chunks
        contextualized_chunks = []
        for original_chunk in original_chunks:
            contextualized_chunk = llm_admin.generate_chunk_context(main_idea=text_main_idea, chunk=original_chunk)
            contextualized_chunks.append(contextualized_chunk)

        # For Pinecone Storing
        # 1. Generate embeddings
        embeddings = []
        for chunk in contextualized_chunks:
            embedding = embedding_admin.generate_embeddings(chunk)
            embeddings.append(embedding)

        # 2. Store embeddings in Pinecone
        pinecone_admin.start_ingestion_process_pinecone(chunks=contextualized_chunks, embeddings=embeddings)


        # For TF-IDF Storing
        tfidf_admin.start_ingestion_process_tfidf(chunks=contextualized_chunks)

        return {"message": "Ingestion process completed"}

    except Exception as e:
        return {"error": f"Error when trying to open the file {pdf_path}: {e}"}