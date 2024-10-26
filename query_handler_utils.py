import json
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
from transformers import pipeline
import torch


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)["questions"][:6] #Remove the slicing
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

if __name__ == '__main__':
    pass
