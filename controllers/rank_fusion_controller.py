from collections import defaultdict
from typing import List, Dict, Any

class RankFusionController:
    def __init__(self, k: float = 60.0):
        self.k = k

    def reciprocal_rank_fusion(self, rankings: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]: 
        fusion_scores = {}

        for rank_list in rankings:
            for position, item in enumerate(rank_list):
                item_text = item['text']

                # RRF formula: 1 / (k + r) where r is the rank position
                score = 1.0 / (self.k + position)

                if item_text in fusion_scores:
                    fusion_scores[item_text]['rrf_score'] += score
                    # Keep track of original scores and metadata
                    fusion_scores[item_text]['sources'].append({
                        'score': item.get('score', 0.0),
                        'metadata': {'text': item.get('text', '')}
                    })
                else:
                    fusion_scores[item_text] = {
                        'rrf_score': score,
                        'sources': [{
                            'score': item.get('score', 0.0),
                            'metadata': {'text': item.get('text', '')}
                        }]
                    }

        # Create final ranked list
        ranked_results = []

        for item_text, data in fusion_scores.items():
            combined_metadata = defaultdict(list)
            for source in data['sources']:
                for key, value in source['metadata'].items():
                    combined_metadata[key].append(value)
            
            ranked_results.append({
                'text': item_text,
                'rrf_score': data['rrf_score'],
                'original_scores': [source['score'] for source in data['sources']],
                'metadata': combined_metadata
            })
        
        ranked_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return ranked_results
    

def format_pinecone_results(pinecone_results: List[Dict]) -> List[Dict]:
    """Format Pinecone results to standard format"""
    return [
        {
            'id': result['id'],
            'score': result['score'],
            'text': result['metadata']['text']
        }
        for result in pinecone_results
    ]

def format_tfidf_results(tfidf_results: List[tuple]) -> List[Dict]:
    """Format TF-IDF results to standard format"""
    return [
        {
            'id': result['id'],
            'score': result['score'],
            'text': result['text']
        }
        for result in tfidf_results
    ]
        