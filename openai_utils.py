from openai import OpenAI, RateLimitError, APIError, Timeout
import config
import time
import tiktoken
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create the client instance with the API key
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    )

def truncate_text(text, max_tokens, encoding_name='gpt-3.5-turbo'):
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def generate_ideal_answer(question_body, snippets):
    max_prompt_tokens = 3500 #Adjust based on model's max context length, 4096 tokens for GPT-3.5-turbo

    # Truncate snippets to prevent exceeding token limits
    snippets_text = '\n'.join(snippets)
    snippets_text = truncate_text(snippets_text,max_prompt_tokens)

    prompt = (
        f"Question: {question_body}\n"
        f"Relevant snippets:\n{snippets}\n"
        "Please provide a concise and comprehensive answer to the question based on these snippets."
    )

    system_prompt = (
        "You are a knowledgeable assistant specializing in biomedical questions."
        "Provide evidence-based, concise answers using the relevant snippets. "
        "Cite relevant findings where appropriate."
    )

    
    for attempt in range(3):  # Retry up to 3 times
        try:
            # Use client to call ChatCompletion with gpt-3.5-turbo model
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.GPT_MAX_TOKENS,
                temperature=config.GPT_TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
        
        except (RateLimitError, APIError, Timeout) as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Error '{e}' on attempt {attempt + 1}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        except Exception as e:
            # Handle unexpected exceptions
            logger.error(f"Unexpected error: {e}")
            raise e

    # If all attempts fail, raise an exception
    raise Exception("Failed to generate answer after multiple attempts due to API errors.")

def generate_exact_answer(question, snippets, question_type):
    
    system_prompt = (
        "You are a specialized biomedical question answering system. "
        "For factoid questions, provide only the specific entity or value asked for. "
        "For list questions, provide a semicolon-separated list of unique items. "
        "For yes/no questions, respond only with 'Yes' or 'No'. "
        "For summary questions, indicate that an exact answer is not appropriate."
    )
    
    prompt = (
        f"Question Type: {question_type}\n"
        f"Question: {question}\n"
        f"Relevant snippets:\n{snippets}\n"
        "Provide the exact answer based on the question type and evidence."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Shorter for exact answers
            temperature=0.3  # Lower temperature for more focused answers
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating exact answer: {e}")
        return "Error generating exact answer"
    
if __name__ == "__main__":
    pass