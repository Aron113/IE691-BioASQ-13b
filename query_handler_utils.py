import json
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
import torch
from openai import OpenAI
import config

def parse_json(file_path):
    """
    Parses a JSON file containing questions and returns the questions data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)["questions"]
    return data

def extract_keywords_baseline(question):
    """
    Extracts keywords from a question using a bag-of-words approach.
    """
    vectorizer = CountVectorizer(stop_words='english')
    keywords = vectorizer.build_tokenizer()(question.lower())
    return list(set(keywords))  # Remove duplicates

def extract_keywords_spacy(question):
    """
    Extracts biomedical terms from a question using SpaCy's `en_core_sci_lg` model.
    """
    spacy_model = spacy.load("en_core_sci_lg")  # SpaCy model for biomedical text
    doc = spacy_model(question)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_keywords_bert(question):
    """
    Extracts biomedical terms using the BioBERT model for Named Entity Recognition (NER).
    """
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Create a pipeline for NER
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Process the text to extract keywords
    results = ner_pipeline(question)

    # Extract keywords and their labels
    extracted_keywords = [(res['word'], res['entity_group']) for res in results]
    return extracted_keywords

def extract_keywords_gpt(question, api_key=config.OPENAI_API_KEY, model="gpt-4"):
    """
    Extracts key biomedical terms from a question using OpenAI's GPT API.
    """
    prompt = f"""
    Extract the key terms from the following question:
    "{question}"

    These terms will be used in an API request to NCBI's Pubmed library to retrieve relevant articles,
    so be sure to extract the relevant biomedical terms and question terms.

    Provide the terms as a JSON array.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )

        # Extract and format the response content
        keywords_content = response.choices[0].message.content
        if keywords_content.startswith("```json") and keywords_content.endswith("```"):
            keywords_content = keywords_content[7:-3].strip()  # Remove JSON code block markers
        return [(keyword, 1) for keyword in json.loads(keywords_content)]
    except Exception as e:
        print(f"Error extracting keywords with GPT: {e}")
        return []

def identify_question_type(question):
    """
    Identifies the type of a biomedical question using rule-based classification.
    """
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
    """
    Extracts an exact answer from snippets based on the question type.
    """
    nlp = spacy.load("en_core_sci_lg")

    if question_type == 'yes_no':
        # Classify as 'Yes' or 'No' based on positive/negative indicators in the snippets
        positive_indicators = sum(1 for s in snippets if any(pos in s.lower()
                                                             for pos in ['confirm', 'prove', 'demonstrate', 'show', 'indicate']))
        negative_indicators = sum(1 for s in snippets if any(neg in s.lower()
                                                             for neg in ['deny', 'refute', 'disprove', 'reject']))
        return 'Yes' if positive_indicators > negative_indicators else 'No'

    elif question_type in ['factoid', 'list']:
        # Extract named entities from the snippets that match question focus
        question_doc = nlp(question)
        question_entities = set(ent.label_ for ent in question_doc.ents)

        answers = []
        for snippet in snippets:
            doc = nlp(snippet)
            for ent in doc.ents:
                if ent.label_ in question_entities:
                    answers.append(ent.text)

        if question_type == 'factoid':
            return answers[0] if answers else "No exact answer found"
        else:  # List type
            return "; ".join(set(answers)) if answers else "No exact answers found"

    return "Question requires detailed explanation"

def prepare_snippets_for_gpt(snippets):
    """
    Combines the top 5 snippets into a single string for use with GPT.
    """
    return " ".join([snippet['text'] for snippet in snippets[:5]])

if __name__ == '__main__':
    pass