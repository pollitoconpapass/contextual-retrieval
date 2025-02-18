import ollama

class EmbedingController:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        
    def generate_embeddings(self, words):
        response = ollama.embed(model=self.model_name, input=words)
        return response["embeddings"][0]