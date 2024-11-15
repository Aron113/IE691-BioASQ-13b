from rouge_score import rouge_scorer
from bert_score import score
import json

def load_training_ideal_answers(training_data_path):
    with open(training_data_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item["ideal_answer"] for item in data["questions"]}

def compute_rouge_scores(training_ideal_answer, generated_ideal_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(training_ideal_answer, generated_ideal_answer)

def compute_bert_scores(training_ideal_answers, generated_ideal_answers):
    P, R, F1 = score(generated_ideal_answers, training_ideal_answers, lang="en", rescale_with_baseline=True)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }

def evaluate_generated_ideal_answers(generated_data, training_data_path):
    generated_ideal_answers = {item["id"]: item["ideal_answer"] for item in generated_data["questions"]}
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