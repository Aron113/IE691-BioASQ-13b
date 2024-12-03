from rouge_score import rouge_scorer
from bert_score import score
from sklearn.metrics import ndcg_score
import json
import numpy as np
import pytrec_eval
import re

def calc_precision(top10_articles, question_ideal_articles, training_data_path):
    qrel = {"query": {re.search(r'/pubmed/(\d+)', article_url).group(1): 1 for article_url in question_ideal_articles}}
    run = {"query": {article["pmid"] : 1 for article in top10_articles}}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P.10'})
    results = evaluator.evaluate(run)
    return results

def load_training_ideal_answers(training_data_path):
    with open(training_data_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item["ideal_answer"] for item in data["questions"]}

def load_training_exact_answers(training_data_path):
    with open(training_data_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item["exact_answer"] for item in data["questions"] if item["type"] in ["factoid", "yesno", "list"]}

def compute_rouge_scores(training_ideal_answer, generated_ideal_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(training_ideal_answer, generated_ideal_answer)

def compute_bert_score_single(training_ideal_answer, generated_ideal_answer):
    
    refs = [training_ideal_answer]
    cands = [generated_ideal_answer]
    
    # Compute BERT score
    P, R, F1 = score(cands, refs, lang="en", rescale_with_baseline=True)
    
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item(),
    }

def compute_bert_scores(training_ideal_answers, generated_ideal_answers):
    P, R, F1 = score(generated_ideal_answers, training_ideal_answers, lang="en", rescale_with_baseline=True)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }

def evaluate_generated_ideal_answers(generated_data, training_data_path):
    generated_ideal_answers = {item["id"]: item["generated_answer"] for item in generated_data}
    training_ideal_answers = load_training_ideal_answers(training_data_path)
    
    rouge_scores = []

    for question_id, generated_ideal_answer in generated_ideal_answers.items():
        training_ideal_answer = training_ideal_answers.get(question_id, "")[0]
        
        # ROUGE scores
        print(training_ideal_answer)
        print(generated_ideal_answer)
        rouge_score = compute_rouge_scores(training_ideal_answer, generated_ideal_answer)
        rouge_scores.append(rouge_score)

    # calculating average ROUGE scores
    average_rouge = {
        "rouge1": {
            "precision": sum([score['rouge1'].precision for score in rouge_scores]) / len(rouge_scores),
            "recall": sum([score['rouge1'].recall for score in rouge_scores]) / len(rouge_scores),
            "fmeasure": sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        },
        "rouge2": {
            "precision": sum([score['rouge2'].precision for score in rouge_scores]) / len(rouge_scores),
            "recall": sum([score['rouge2'].recall for score in rouge_scores]) / len(rouge_scores),
            "fmeasure": sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        },
        "rougeL": {
            "precision": sum([score['rougeL'].precision for score in rouge_scores]) / len(rouge_scores),
            "recall": sum([score['rougeL'].recall for score in rouge_scores]) / len(rouge_scores),
            "fmeasure": sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        }
    }

    # Compute BERT scores
    reference_list = [training_ideal_answers[qid] for qid in generated_ideal_answers.keys() if qid in training_ideal_answers]
    generated_list = [generated_ideal_answers[qid] for qid in generated_ideal_answers.keys() if qid in training_ideal_answers]
    
    bert_score_avg = compute_bert_scores(reference_list, generated_list)
    
    return {
        "average_rouge": average_rouge,
        "average_bert": bert_score_avg
    }

def evaluate_generated_exact_answers(generated_data, training_data_path):
    generated_exact_answers = {item["id"]: item["generated_answer"] for item in generated_data}
    training_exact_answers = load_training_exact_answers(training_data_path)
    
    numerator, denominator = 0, 0

    for question_id, generated_exact_answer in generated_exact_answers.items():
        training_exact_answer = training_exact_answers.get(question_id, "")[0]
        
        # ROUGE scores
        print(training_exact_answer)
        print(generated_exact_answer)
        if training_exact_answer == generated_exact_answer:
            numerator += 1
        denominator += 1
    
    if not denominator:
        return 0