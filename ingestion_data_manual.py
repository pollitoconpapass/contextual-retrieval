import os
from dotenv import load_dotenv
from controllers.llm_controller import LLMController
from controllers.embedding_controller import EmbedingController
from controllers.document_reading_controller import DocumentExtractionController
from controllers.document_processing_controller import PineconeController, TFIDFController

load_dotenv('./.env')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
START_PAGE = int(os.getenv("START_PAGE"))
END_PAGE = int(os.getenv("END_PAGE"))

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
        print(f"Starting extraction from {PDF_PATH}")
        print(f"Page range: {START_PAGE} to {END_PAGE}")

        text_extracted = reading.extract_text_from_pdf(start_page=START_PAGE, end_page=END_PAGE)
        print(f"Extracted text length: {len(text_extracted)} characters")

        # Generate chunks
        original_chunks = reading.generate_chunks(text=text_extracted)
        print(f"Number of original chunks: {len(original_chunks)}")

        # Generate main idea
        text_main_idea = llm_admin.generate_main_text_idea(text=text_extracted)

        # Give context to chunks
        count = 0
        contextualized_chunks = []
        for original_chunk in original_chunks:
            contextualized_chunk = llm_admin.generate_chunk_context(main_idea=text_main_idea, chunk=original_chunk)
            contextualized_chunks.append(contextualized_chunk)

            count += 1
            print(f"Contextualized chunk {count}/{len(original_chunks)}")

        # For Pinecone Storing
        # 1. Generate embeddings
        print("\nStarting with the embeddings now...")
        embeddings = []
        for chunk in contextualized_chunks:
            embedding = embedding_admin.generate_embeddings(chunk)
            embeddings.append(embedding)

        # 2. Store embeddings in Pinecone
        print("Storing embeddings in Pinecone...")
        pinecone_admin.start_ingestion_process_pinecone(chunks=contextualized_chunks, embeddings=embeddings)


        # For TF-IDF Storing
        print("\nStoring data in the TF-IDF index...")
        tfidf_admin.start_ingestion_process_tfidf(chunks=contextualized_chunks)

        print("\nIngestion Process Completed")

    except Exception as e:
        print(f"Error when trying to open the file {PDF_PATH}: {e}")

if __name__ == "__main__":
    main_ingestion_process()