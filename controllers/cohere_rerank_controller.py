import cohere

class CohereRerankController:
    def __init__(self, api_key, model_name="rerank-multilingual-v3.0"):
        self.cohere_client = cohere.ClientV2(api_key)
        self.model_name = model_name

    def format_responses(self, responses: list) -> list:
        formatted_responses = [item['metadata']['text'][0] for item in responses]
        return formatted_responses

    def rerank(self, query: str, responses: list, num_responses: int=4):
        formatted_responses = self.format_responses(responses)
        reranked_responses = self.cohere_client.rerank(
            query=query,
            documents=formatted_responses,
            top_n=num_responses,
            model=self.model_name,
            return_documents=True
        )

        reranked_documents = [item.document.text for item in reranked_responses.results]
        return reranked_documents