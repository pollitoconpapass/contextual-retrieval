from rerankers import Reranker

class CustomRerankController:
    def __init__(self, model_name: str, language_code: str):
        self.ranker = Reranker(model_name, lang=language_code, verbose=0)

    def format_responses(self, responses: list) -> list:
        formatted_responses = [item['metadata']['text'][0] for item in responses]
        return formatted_responses

    def rerank(self, query: str, responses: list):
        formatted_responses = self.format_responses(responses)
        rank_result = self.ranker.rank(query, docs=formatted_responses)

        final_results = [result.document.text for result in rank_result.results]
        return final_results
        