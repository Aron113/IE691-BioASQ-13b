import config
import query_handler_utils
import search_utils
import ranking_utils
import openai_utils
import evaluation_utils
from sentence_transformers import SentenceTransformer, util
import torch
import json

def save_results(results):
    with open("phase_b_results.json", "w") as f:
        json.dump({"questions": results}, f, indent=2)


def run(file_path):
    questions = query_handler_utils.parse_json(file_path)
    results = []

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
        # Prepare snippets for ChatGPT
        combined_snippets = query_handler_utils.prepare_snippets_for_gpt(snippet_list)
        
        # Generate the ideal answer using ChatGPT API
        ideal_answer = openai_utils.generate_ideal_answer(question_body, combined_snippets)

        # Collect results for Phase B
        result = {
            "id": question["id"],
            "ideal_answer": ideal_answer
        }
        results.append(result)

    # Save results in the required JSON format for submission
    save_results(results)
    phase_b_evaluation = evaluation_utils.evaluate_generated_ideal_answers(results, file_path)

    return

if __name__ == '__main__':
    training_data_path = config.TRAINING_DATA_PATH
    run(training_data_path)