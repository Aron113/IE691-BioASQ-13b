import config
import query_handler_utils
import search_utils
import ranking_utils
from sentence_transformers import SentenceTransformer, util
import torch

def run(file_path):
    questions = query_handler_utils.parse_json(file_path)
    for question in questions:
        question_body = question["body"]
        question_keywords = query_handler_utils.extract_keywords_spacy(question_body)
        question_keywords = [i[0] for i in question_keywords]
        
        #Search NCBI for relevant articles
        query_term = search_utils.ncbi_querybuilder(question_keywords)
        pmid_list = search_utils.ncbi_query(config.NCBI_RETMAX, query_term, config.MIN_DATE, config.MAX_DATE)

        #Get the pmid, title and abstract of the relevant articles in the form of a list
        article_info_list = search_utils.ncbi_title_abstract_query(pmid_list)

        ########### Phase A ###########
        #Rank the articles on decreasing relevance to the question/query
        model = SentenceTransformer("all-MiniLM-L6-v2")
        articles_ranked_list = ranking_utils.rank_abstract(article_info_list, question_body, model)
        top10_articles = articles_ranked_list[:11]

        #From the top 10 articles, find the most relevant snippet in each article's abstract
        snippet_list = ranking_utils.rank_snippet(top10_articles, question_body, model)


        ########### Phase B ###########

    return

if __name__ == '__main__':
    training_data_path = config.TRAINING_DATA_PATH
    run(training_data_path)