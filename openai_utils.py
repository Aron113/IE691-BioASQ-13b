from openai import OpenAI, RateLimitError
import config
import time

# Create the client instance with the API key
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    )

def generate_ideal_answer(question_body, snippets):
    prompt = (
        f"Question: {question_body}\n"
        f"Relevant snippets:\n{snippets}\n"
        "Please provide a concise and comprehensive answer to the question based on these snippets."
    )

    
    for attempt in range(3):  # Retry up to 3 times
        try:
            # Use client to call ChatCompletion with gpt-3.5-turbo model
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant for biomedical questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.GPT_MAX_TOKENS,
                temperature=config.GPT_TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
        
        except RateLimitError:
            # Handle rate limit exceeded by waiting before retrying
            print(f"Rate limit exceeded on attempt {attempt + 1}. Retrying in 20 seconds...")
            time.sleep(20)  # Wait 20 seconds before retrying

    # If all attempts fail, raise an exception
    raise Exception("Failed to generate answer after multiple attempts due to rate limits.")