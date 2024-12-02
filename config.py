from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

TRAINING_DATA_PATH = 'BioASQ-training13b/training13b.json'
NCBI_RETMAX = 30 #Maximum number of articles that are retrieved in the API call from NCBI
MIN_DATE = '2000/01/01'
MAX_DATE = '2025/01/01'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MAX_TOKENS = 300  # Set to the desired max tokens for ChatGPT responses
GPT_TEMPERATURE = 0.5  # Adjust temperature to control response creativity

BASELINE_TOP_SNIPPETS = 5  # Number of snippets to include in the generated answer