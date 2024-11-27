from sentence_transformers import SentenceTransformer, util
import torch

def rank_abstract(article_info_list, question_body, model):
    model = model

    #Represent the question and article abstracts as embeddings
    question_embedding = model.encode(question_body)
    articles_embeddings = model.encode([article['abstract'] for article in article_info_list])
    similarity_scores = util.pytorch_cos_sim(question_embedding, articles_embeddings)

    #Sort the scores in descending order along with their indices
    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
    sorted_indices = sorted_indices.flatten()

    # Reorder the articles based on the sorted indices
    sorted_articles = [article_info_list[i] for i in sorted_indices]
    return sorted_articles

def find_snippet_location(article, snippet_str):
    snip_length = len(snippet_str)
    start_ind = article['abstract'].find(snippet_str)
    if start_ind != -1: #Snippet can be found in abstract
        return 'abstract', start_ind, start_ind + snip_length
            
    start_ind = article['title'].find(snippet_str)
    if start_ind != -1: #Snippet can be found in title
        return 'title', start_ind, start_ind + snip_length

def rank_snippet(top10_articles, question_body, model):

    snippet_list = []
    for article in top10_articles:
        if article['abstract']:
            abstract = article['abstract']
            abstract_sentences = abstract.split('. ')

            if article['title']:
                abstract_sentences.append(article['title'])

            query_embedding = model.encode(question_body, normalize_embeddings=True)
            corpus_embeddings = model.encode(abstract_sentences, normalize_embeddings=True)
            dot_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(dot_scores, k=min(3, len(dot_scores))) # Get up to 3 sentences
            if top_results.indices.size(0) > 0:  # Ensure there are valid results
                snippet_str = abstract_sentences[top_results.indices[0]] # Get the most similar sentence/snippet
                # Continue processing the snippet
                section, start, end = find_snippet_location(article, snippet_str)
                snip = {
                    'pmid': article['pmid'],
                    'offsetInBeginSection': start,
                    'offsetInEndSection': end,
                    'beginSection': section,
                    'endSection': section,
                    'text': snippet_str
                 }
                snippet_list.append(snip)
            
    return snippet_list


if __name__ == '__main__':
    pass