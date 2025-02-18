import os
from dotenv import load_dotenv
from controllers.llm_controller import LLMController
from controllers.embedding_controller import EmbedingController
from controllers.document_reading_controller import DocumentExtractionController
from controllers.document_processing_controller import PineconeController, TFIDFController

load_dotenv('./.env')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")

TFIDF_INDEX_NAME = os.getenv("TFIDF_INDEX_NAME")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL_NAME")


def main_ingestion_process():
    try: 
        # === DECLARE ALL THE OBJECTS ===
        reading = DocumentExtractionController(pdf_path=PDF_PATH)
        embedding_admin = EmbedingController(model_name=EMBEDDING_MODEL)
        llm_admin = LLMController(model_name=LLM_MODEL)
        pinecone_admin = PineconeController(pinecone_api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
        tfidf_admin = TFIDFController(TFIDF_INDEX_NAME)


        # Extracting Text from the PDF doc
        text_extracted = reading.extract_text_from_pdf()
        full_text = " ".join(text_extracted)

        # Generate chunks
        original_chunks = reading.generate_chunks(text=full_text)
        print(f"Number of original chunks: {len(original_chunks)}")

        # Generate main idea
        text_main_idea = llm_admin.generate_main_text_idea(text=full_text)

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

        print("Ingestion Process Completed")

    except Exception as e:
        print(f"Error when trying to open the file {PDF_PATH}: {e}")

if __name__ == "__main__":
    main_ingestion_process()