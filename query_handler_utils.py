import json
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
from transformers import pipeline
import torch


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)["questions"]
    return data

def extract_keywords_spacy(question):
    spacy_model = spacy.load("en_core_sci_lg") #Spacy model for processing biomedical text, https://allenai.github.io/scispacy/
    doc = spacy_model(question)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_keywords_bert(question):
    # # Load PubMedBERT model and tokenizer
    # model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    # inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    # outputs = model(**inputs, output_attentions=True)
    # attentions = outputs.attentions[-1].squeeze(0)  # Take the last layer's attentions

    # # Aggregate attention scores across heads
    # avg_attentions = attentions.mean(dim=0).detach().numpy()
    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    # # Identify high-attention tokens
    # keywords = [tokens[i] for i, score in enumerate(avg_attentions.flatten()) if score > 0.2 and tokens[i] not in ["[CLS]", "[SEP]"]]

    # # Join tokens that belong to the same word
    # refined_keywords = []
    # for token in keywords:
    #     if token.startswith("##"):
    #         refined_keywords[-1] += token[2:]
    #     else:
    #         refined_keywords.append(token)
    # return refined_keywords
    
    # Load BioBERT model and tokenizer
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Create a pipeline for NER
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Process the text to extract keywords
    results = ner_pipeline(question)

    # Display the extracted keywords and their labels
    extracted_keywords = [(res['word'], res['entity_group']) for res in results]

    return extracted_keywords

def identify_question_type(question): #Rule-based classificiation to identify the type of biomedical question (factoid, list, yes/no, summary).
    
    question_lower = question.lower()
    
    # Pattern matching for different question types
    if any(word in question_lower for word in ['what', 'which', 'who', 'where', 'when']):
        if re.search(r'list|name|mention|identify', question_lower):
            return 'list'
        return 'factoid'
    elif question_lower.startswith(('is', 'are', 'does', 'do', 'can', 'could', 'will', 'would')):
        return 'yes_no'
    elif any(word in question_lower for word in ['how', 'explain', 'describe', 'why']):
        return 'summary'
    return 'summary'  # Default to summary for complex questions

def extract_exact_answer(question, snippets, question_type):
    
    nlp = spacy.load("en_core_sci_lg")
    
    if question_type == 'yes_no':
        # Implement yes/no classification based on snippets
        positive_indicators = sum(1 for s in snippets if any(pos in s.lower() 
                                for pos in ['confirm', 'prove', 'demonstrate', 'show', 'indicate']))
        negative_indicators = sum(1 for s in snippets if any(neg in s.lower() 
                                for neg in ['deny', 'refute', 'disprove', 'reject']))
        return 'Yes' if positive_indicators > negative_indicators else 'No'
    
    elif question_type in ['factoid', 'list']:
        # Extract named entities relevant to the question
        question_doc = nlp(question)
        question_entities = set(ent.label_ for ent in question_doc.ents)
        
        answers = []
        for snippet in snippets:
            doc = nlp(snippet)
            for ent in doc.ents:
                # Match entity types with question focus
                if ent.label_ in question_entities:
                    answers.append(ent.text)
        
        if question_type == 'factoid':
            return answers[0] if answers else "No exact answer found"
        else:  # list type
            return "; ".join(set(answers)) if answers else "No exact answers found"
    
    return "Question requires detailed explanation"

def prepare_snippets_for_gpt(snippets):
    # Example: Combine the top 5 snippets into a single string for ChatGPT prompt
    combined_snippets = " ".join([snippet['text'] for snippet in snippets[:5]])
    return combined_snippets

if __name__ == '__main__':
    pass
