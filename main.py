import config
import query_handler_utils
import search_utils
import ranking_utils
import openai_utils
import evaluation_utils
from sentence_transformers import SentenceTransformer
import json
import re


def save_results(results, output_file="results.json"):
    """
    Saves the results in a JSON file.
    """
    with open(output_file, "w") as f:
        json.dump({"questions": results}, f, indent=2)


def run_baseline(file_path):
    """
    Runs the baseline pipeline for question answering.
    """
    questions = query_handler_utils.parse_json(file_path)
    results = []
    ground_truth_ideal_answers = evaluation_utils.load_training_ideal_answers(file_path)

    for question in questions[:10]:  # Limit to 10 questions for testing
        question_body = question["body"]
        question_id = question["id"]

        # Step 1: Keyword Extraction
        keywords = query_handler_utils.extract_keywords_baseline(question_body)

        # Step 2: Query Construction and Article Retrieval
        query_term = search_utils.construct_query_baseline(keywords)
        pmid_list = search_utils.ncbi_query(config.NCBI_RETMAX, query_term, config.MIN_DATE, config.MAX_DATE)
        articles = search_utils.ncbi_title_abstract_query(pmid_list)

        if not articles:
            print(f"No articles found for question {question_id}")
            continue

        # Step 3: Snippet Selection
        snippets = ranking_utils.select_snippets_baseline(articles, keywords)

        # Step 4: Generate Baseline Answer
        combined_snippets = ' '.join(snippets[:config.BASELINE_TOP_SNIPPETS])
        generated_answer = combined_snippets

        # Step 5: Evaluate Generated Answer
        ground_truth_answer = ground_truth_ideal_answers.get(question_id, [""])[0]
        rouge_score = evaluation_utils.compute_rouge_scores(ground_truth_answer, generated_answer)
        bert_score = evaluation_utils.compute_bert_score_single(ground_truth_answer, generated_answer)

        # Add results to the output
        result = {
            "id": question_id,
            "question": question_body,
            "generated_answer": generated_answer,
            "rouge_score": rouge_score,
            "bert_score": bert_score,
        }

        results.append(result)
        print(f"Question ID: {question_id}, ROUGE Score: {rouge_score}, BERT Score: {bert_score}")

    # Save results for baseline
    save_results(results, output_file="baseline_results.json")


def run_advanced(file_path):
    """
    Runs the advanced pipeline for question answering, including snippet ranking and GPT-generated answers.
    """
    questions = query_handler_utils.parse_json(file_path)
    results = []
    ground_truth_ideal_answers = evaluation_utils.load_training_ideal_answers(file_path)
    num_qns = 0
    total_precision = 0

    for question in questions:
        num_qns += 1
        
        question_body = question["body"]
        question_id = question["id"]
        question_type = question.get("type","ideal")

        # Step 1: Keyword Extraction
        question_keywords = query_handler_utils.extract_keywords_spacy(question_body)
        question_keywords = [i[0] for i in question_keywords]
        print(f"Extracted Keywords: {question_keywords}")

        # Step 2: Query Construction and Article Retrieval
        query_term = search_utils.ncbi_querybuilder(question_keywords)
        pmid_list = search_utils.ncbi_query(config.NCBI_RETMAX, query_term, config.MIN_DATE, config.MAX_DATE)
        article_info_list = search_utils.ncbi_title_abstract_query(pmid_list)
        if not article_info_list:
            print(f"No articles found for question {question_id}")
            continue

        # Step 3: Article Ranking
        model = SentenceTransformer("all-MiniLM-L6-v2")
        articles_ranked_list = ranking_utils.rank_abstract(article_info_list, question_body, model)
        top10_articles = articles_ranked_list[:10]

        # Precision Evaluation for Top Articles
        question_ideal_articles = question.get("documents", [])
        eval_results = evaluation_utils.calc_precision(top10_articles, question_ideal_articles, file_path)
        print(f"Precision@10 for Question {question_id}: {eval_results}")

        total_precision += eval_results
        average_precision = total_precision / num_qns
        print(f"Average Precision: {average_precision} , Number of Questions: {num_qns}")

        # Step 4: Snippet Ranking
        snippet_list = ranking_utils.rank_snippet(top10_articles, question_body, model)

        # Step 5: Generate Ideal Answer using GPT
        combined_snippets = query_handler_utils.prepare_snippets_for_gpt(snippet_list)
        if question_type in ["factoid", "yesno", "list"]:
            generated_answer = openai_utils.generate_exact_answer(question_body, combined_snippets, question_type)
        else:
            generated_answer = openai_utils.generate_ideal_answer(question_body, combined_snippets)

        # Collect Results for Advanced Pipeline
        result = {
            "id": question["id"],
            "type": question["type"],
            "question": question["body"],
            "generated_answer": generated_answer
        }
        print(f"Question: {question_body}, Type: {question_type}")
        print(f"Generated Answer for Question {question_id}: {generated_answer}")

        # Step 6: Evaluate Generated Answer
        ground_truth_answer = ground_truth_ideal_answers.get(question_id, [""])[0]
        rouge_score = evaluation_utils.compute_rouge_scores(ground_truth_answer, generated_answer)
        bert_score = evaluation_utils.compute_bert_score_single(ground_truth_answer, generated_answer)
        print(f"ROUGE Scores for Question {question_id}: {rouge_score}")
        print(f"BERT Scores for Question {question_id}: {bert_score}")

        results.append(result)

    # Save results for advanced pipeline
    save_results(results, output_file="advanced_results.json")
    phase_b_evaluation = evaluation_utils.evaluate_generated_ideal_answers(results, file_path)
    print(f"Phase B Evaluation: {phase_b_evaluation}")


if __name__ == "__main__":
    training_data_path = config.TRAINING_DATA_PATH

    # Run Baseline Model
    print("Running Baseline Pipeline...")
    run_baseline(training_data_path)

    # Run Advanced Model
    print("Running Advanced Pipeline...")
    run_advanced(training_data_path)