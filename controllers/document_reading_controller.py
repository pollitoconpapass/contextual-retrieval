import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentExtractionController:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self, start_page: int = 0, end_page: int = None):
        try:
            with open(self.pdf_path, 'rb') as pdf_file_obj:
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                num_pages = len(pdf_reader.pages)
                text = []

                if end_page is None or end_page > num_pages:
                    end_page = num_pages

                page_range = range(start_page, end_page)

                if start_page >= num_pages:
                    raise ValueError(f"Start page {start_page} exceeds PDF length of {num_pages} pages")

                if start_page > end_page:
                    raise ValueError("Start page number cannot be greater than end page number.")
                
                for page_num in page_range:
                    page_obj = pdf_reader.pages[page_num]
                    page_text = page_obj.extract_text()
                    if page_text.strip():
                        text.append(page_text)
                    else:
                        print(f"Unable to extract text from page {page_num + 1} of {self.pdf_path}.")

                print(f"Text extracted from {num_pages} pages")
                
            return "\n\n".join(text)
        
        except Exception as e:
            print(f"Error when trying to open the file {self.pdf_path}: {e}")
            return []
        

    def generate_chunks(self, text: str)-> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )   

        chunks = text_splitter.split_text(text)
        return chunks
    