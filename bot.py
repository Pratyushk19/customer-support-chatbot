import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify
import logging
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Ensure the API key is passed as a command-line argument
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Configure the Gemini API with the API key
genai.configure(api_key=api_key)

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

# Define the predefined website URL
WEBSITE_URL = "https://www.saucedemo.com/inventory.html"

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching text from URL: {url} - {e}", exc_info=True)
        return ""

# Function to extract all pages within a website
def extract_all_pages_from_website(url):
    base_url = urlparse(url).scheme + "://" + urlparse(url).netloc
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a", href=True)
        pages = []
        for link in tqdm(links, desc="Fetching pages", unit="page"):
            href = link["href"]
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            page_text = extract_text_from_url(href)
            if page_text:
                pages.append(page_text)
        return " ".join(pages)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching pages from website: {url} - {e}", exc_info=True)
        return ""

# Function to generate a response using the Gemini API
def generate_response_gemini(query, context, max_tokens):
    try:
        context_length = min(13000, len(context))
        prompt = f"Context: {context[:context_length]}...\n\n{query}"
        logging.info(f"Sending request to Gemini API with prompt: {prompt[:100]}...")  # Log first 100 chars of prompt
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=1,
                top_k=50,
                max_output_tokens=max_tokens,
            )
        )
        logging.info(f"Generated response: {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error generating response: {e}", exc_info=True)
        return f"Error generating response: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape_website', methods=['GET'])
def scrape_website():
    website_text = extract_all_pages_from_website(WEBSITE_URL)
    return jsonify({'website_text': website_text})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    website_text = data.get('website_text', '')
    max_tokens = 2048
    response = generate_response_gemini(user_input, website_text, max_tokens=max_tokens)
    return jsonify({'answer': str(response)})

if __name__ == "__main__":
    app.run(debug=True)