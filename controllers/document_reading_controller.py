import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentExtractionController:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self):
        try:
            with open(self.pdf_path, 'rb') as pdf_file_obj:
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                num_pages = len(pdf_reader.pages)
                text = []
                
                for page in range(num_pages):
                    page_obj = pdf_reader.pages[page]
                    page_text = page_obj.extract_text()
                    if page_text:
                        text.append(page_text)
                    else:
                        print(f"Unable to extract text from page {page + 1} of {self.file_path}.")
                print(f"Text extracted from {num_pages} pages")
            return text 
        
        except Exception as e:
            print(f"Error when trying to open the file {self.file_path}: {e}")
            return []
        

    def generate_chunks(self, text: str)-> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )   

        chunks = text_splitter.split_text(text)
        return chunks
    