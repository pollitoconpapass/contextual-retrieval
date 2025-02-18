import ollama

DOCUMENT_CONTEXT_PROMPT = """
Analyze the text and extract the main idea
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Given the main idea of the document:
<main_idea>
{main_idea}
</main_idea>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only in ENGLISH with the succinct context and nothing else.
"""

LLM_ASSISTANT_PROMPT = """
You are a helpful AI assistant who will answer the user's question. Based on the context given. 
<context>
{context}
</context>

<question>
{question}
</question>
"""

class LLMController:
    def __init__(self, model_name="dolphin-mistral"):
        self.model_name = model_name

    def truncate_text(self, text, max_size=12000):
        if len(text) > max_size:
            return text[:max_size]
        
        return text

    def generate_main_text_idea(self, text):
        truncated_text = self.truncate_text(text)

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": DOCUMENT_CONTEXT_PROMPT.format(doc_content=truncated_text)},
            ]
        )

        return response['message']['content']
    
    def generate_chunk_context(self, main_idea, chunk):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": CHUNK_CONTEXT_PROMPT.format(main_idea=main_idea, chunk_content=chunk)},
            ]
        )

        return response['message']['content']
    
    def chat_llm(self, context, message):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": LLM_ASSISTANT_PROMPT.format(context=context, question=message)},
            ]
        )

        return response['message']['content']