import json
import logging
import time
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec,PineconeException
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from flask_cors import CORS
import threading
import time
from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import functools
import random
from datetime import datetime, timedelta


load_dotenv()


# Initialize Flask app
app = Flask(__name__)

CORS(app)

cache = {}
CACHE_TTL = timedelta(hours=6)

# Set up logging
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Pinecone
# Initialize Pinecone client
pinecone = Pinecone(
    api_key= os.environ.get("PINECONE_API_KEY")  # Replace with your API key
)

# Define the index name and check if it exists
index_name = "daawo-vectordb"

# Connect to the index
index = pinecone.Index(index_name)

# Check connection by describing the index stats
print(index.describe_index_stats())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize HuggingFace embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay + random.uniform(0, 0.5))  # slight jitter
                    delay *= backoff_factor
        return wrapper_retry
    return decorator_retry

@retry_with_backoff(max_retries=3, initial_delay=1)
def safe_pinecone_query(index, query_embedding, top_k, filters):
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )

@retry_with_backoff(max_retries=3, initial_delay=1)
def safe_pinecone_upsert(index, vectors):
    index.upsert(vectors=vectors)



def extract_keywords_from_query(query):
    # Remove punctuation and lowercase
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    words = cleaned.split()

    # Remove short/common words
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 3]
    return keywords

# Function to clean dataset-specific content
def clean_output(raw_text):
    """
    Refines and cleans text for LLM context in a RAG system.
    Args:
        raw_text (str): The raw text to be cleaned.
    Returns:
        str: The refined text.
    """
    # Step 1: Remove artifacts (timestamps, encoding errors)
    cleaned_text = re.sub(r"\d{2}/\d{2}/\d{2,4}.*\n", "", raw_text)  # Remove timestamps
    cleaned_text = cleaned_text.replace("™", "'").replace("ﬂ", "").replace("ﬁ", "fi")  # Fix encoding artifacts
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)  # Remove extra spaces

    # Step 2: Consolidate sections
    cleaned_text = re.sub(r"K EY ME DIC INA L PL A N T S", "**Key Medicinal Plants**", cleaned_text)
    cleaned_text = re.sub(r"K e y Constituents", "**Key Constituents**", cleaned_text)
    cleaned_text = re.sub(r"K e y Ac tions", "**Key Actions**", cleaned_text)
    cleaned_text = re.sub(r"R ese arc h", "**Research**", cleaned_text)
    cleaned_text = re.sub(r"Traditional & C urrent Uses", "**Traditional & Current Uses**", cleaned_text)

    # Step 3: Standardize bullet points
    cleaned_text = re.sub(r"- ", "- ", cleaned_text)  # Standardize single dashes as bullet points
    cleaned_text = re.sub(r"\s*-\s+", "- ", cleaned_text)  # Fix bullet point spacing
    
    # Remove truncation indicators and normalize
    cleaned_text = re.sub(r"has$", "has been known to", cleaned_text)

    # Step 4: Remove redundant lines
    lines = cleaned_text.split("\n")
    seen_lines = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)

    # Step 5: Format final output
    refined_text = "\n".join(unique_lines)

    return refined_text

# Function to clean general metadata
def clean_metadata(metadata):
    """
    Clean and normalize metadata fields.
    Args:
        metadata (dict): The metadata dictionary.
    Returns:
        dict: The cleaned and normalized metadata.
    """
    # Normalize remedy_name or name
    if "remedy_name" in metadata:
        metadata["remedy_name"] = metadata["remedy_name"].replace("_", " ").title()
    if "name" in metadata:
        metadata["name"] = metadata["name"].replace("_", " ").title()

    # Clean content/description
    if "remedy_description" in metadata:
        metadata["remedy_description"] = re.sub(r"\s{2,}", " ", metadata["remedy_description"]).strip()
    if "description" in metadata:
        metadata["description"] = re.sub(r"\s{2,}", " ", metadata["description"]).strip()

    return metadata


# Function to query the vector database
def query_vector_db(query, top_k=5, filters=None):
    """
    Query the Pinecone vector database with optional metadata filters.
    Args:
        query (str): The search query.
        top_k (int): Number of results to retrieve.
        filters (dict): Metadata filters for the query (optional).
    Returns:
        list: List of search results with metadata and scores.
    """
    if not query.strip():
        return []
    
    # Generate query embedding
    query_embedding = hf_embeddings.embed_query(query)
    
    
    search_results = safe_pinecone_query(index, query_embedding, top_k, filters)

    # Process and clean results
    results = []
    for match in search_results["matches"]:
        metadata = match["metadata"]

        # Simplify IDs
        if match["id"].startswith("herbs-") or match["id"].startswith("herb-") or match["id"].startswith("herb-and_"):
            match["id"] = match["id"].replace("herb-and_", "")
            match["id"] = match["id"].replace("herbs-", "")
            match["id"] = match["id"].replace("herb-", "")

        # Normalize subcategories
        if "subcategory" in metadata:
            metadata["subcategory"] = metadata["subcategory"].replace("_", " ").title()

        # Standardize units in descriptions
        if "description" in metadata or "remedy_description" in metadata:
            for key in ["description", "remedy_description"]:
                if key in metadata:
                    metadata[key] = re.sub(r"\b(\d+)\s*g\b", r"\1 grams", metadata[key])

        # Detect and handle truncation in descriptions
        if metadata.get("description", "").endswith("has"):
            metadata["description"] += " additional benefits not listed here."

        # Apply dataset-specific cleaning if "page_number" exists
        if metadata.get("page_number") is not None:
            metadata["content"] = clean_output(metadata["content"])
        else:  # General cleaning for other datasets
            metadata = clean_metadata(metadata)

        # Append processed result
        results.append({
            "id": match["id"],
            "score": match["score"],
            "metadata": metadata
        })


    return results


# Function to interact with LLM
def herbal_medicine_query_with_context(query, context_chunks):
    context = "\n\n".join(
        [
            chunk["metadata"].get("description") or
            chunk["metadata"].get("remedy_description") or
            chunk["metadata"].get("content", "")
            for chunk in context_chunks
        ]
    )

    prompt = f"""
    Answer the user's question using the context provided.

    Context:
    {context}

    Question: {query}
    """

    print("Sending prompt to OpenAI...")
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful herbal medicine assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content
        print(f"Response received in {time.time() - start:.2f} seconds")
        print(f"LLM Prompt:\n{prompt}")
        print(f"LLM Raw Response: {response_text}")

    except Exception as e:
        logging.error(f"OpenAI API call failed: {str(e)}")
        return {
            "main_answer": "An error occurred during LLM processing.",
            "additional_notes": str(e)
        }

    # Extract main answer and additional notes
    main_answer = ""
    additional_notes = ""
    try:
        if "Answer:" in response_text:
            response_text = response_text.split("Answer:", 1)[-1].strip()
        
        if "Additional Notes:" in response_text:
            main_answer, additional_notes = response_text.split("Additional Notes:", 1)
            main_answer = main_answer.strip()
            additional_notes = additional_notes.strip()
        else:
            main_answer = response_text.strip()
            additional_notes = "No additional notes provided."
    except Exception as e:
        main_answer = "An error occurred while parsing the response."
        additional_notes = str(e)
    
    return {
        "main_answer": main_answer,
        "additional_notes": additional_notes
    }


def fetch_relevant_context(query, top_k=5, relevance_score_threshold=0.75):
    """
    Fetch relevant context chunks from the vector database for a given query.
    Args:
        query (str): The user query.
        top_k (int): Number of top results to retrieve.
    Returns:
        list: Relevant context chunks.
    """
    # Step 1: Extract keywords
    keywords = extract_keywords_from_query(query)
    keyword_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b', re.IGNORECASE)

    # Step 2: Get initial top-K results
    context_chunks = query_vector_db(query, top_k=top_k * 2)  # Get more for filtering

    # Step 3: Filter chunks by keyword match or similarity score
    filtered_chunks = []
    for chunk in context_chunks:
        text = chunk["metadata"].get("description") or chunk["metadata"].get("remedy_description") or chunk["metadata"].get("content", "")
        score = chunk["score"]

        keyword_match = keyword_pattern.search(text)
        if keyword_match or score >= relevance_score_threshold:
            filtered_chunks.append(chunk)

    # Step 4: Sort and return final filtered top-k
    final_chunks = sorted(filtered_chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    print(f"Selected {len(final_chunks)} filtered context chunks from {len(context_chunks)} candidates.")
    return final_chunks

def full_pipeline_test(query):
    """
    Test the full pipeline: context retrieval + LLM query.
    Args:
        query (str): User query.
    """

    print("Fetching context...")
    context = fetch_relevant_context(query)
    print("Context fetched. Now querying LLM...")

    response = herbal_medicine_query_with_context(query, context)

    print("LLM responded.")
    return response


# Flask Routes
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Herbal Medicine API!"})

@app.route("/query", methods=["POST"])
def query_endpoint():
    try:
        print(f"Incoming {request.method} request to {request.path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Payload: {request.get_data(as_text=True)}")
        data = request.get_json(force=True)
        query = data.get("query", "")
        
        # Add before calling fetch_relevant_context or full_pipeline_test
        query = query.strip().lower()
        if not query or len(query) == 0:
            logging.warning(f"Rejected query: '{query}'")
            return jsonify({"error": "Query is empty or only contains whitespace."}), 400
        if len(query) > 300:
            logging.warning(f"Rejected query: '{query}'")
            return jsonify({"error": "Query exceeds the maximum allowed length of 300 characters."}), 400
        
        cached_entry = cache.get(query)

        if cached_entry:
            age = datetime.now() - cached_entry["timestamp"]
            if age < CACHE_TTL:
                logging.info("Cache HIT")
                return jsonify({
                    "main_answer": cached_entry["main_answer"],
                    "additional_notes": cached_entry["additional_notes"],
                    "cached": True
                })
            else:
                logging.info("Cache EXPIRED")
                cache.pop(query)
        else:
            logging.info("Cache MISS")

        context = fetch_relevant_context(query)
        llm_response = herbal_medicine_query_with_context(query, context)

        # Store in cache
        cache[query] = {
            "main_answer": llm_response["main_answer"],
            "additional_notes": llm_response.get("additional_notes", "No additional notes provided."),
            "timestamp": datetime.now()
        }

        return jsonify(llm_response)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Logs full stack trace
        
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": F"Internal server error: {str(e)}"}), 500

@app.route("/pipeline-test", methods=["POST"])
def full_pipeline():
    print(f"Incoming {request.method} request to {request.path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Payload: {request.get_data(as_text=True)}")

    try:
        data = request.get_json()
        query = data.get("query", "")
        print("----------------- Testing -------------------", query)
        
        query = query.strip()
        if not query or len(query) == 0:
            logging.warning(f"Rejected query: '{query}'")
            return jsonify({"error": "Query is empty or only contains whitespace."}), 400
        if len(query) > 300:
            logging.warning(f"Rejected query: '{query}'")
            return jsonify({"error": "Query exceeds the maximum allowed length of 300 characters."}), 400
        
        response = full_pipeline_test(query)
        return jsonify({
            "query": query,
            "response": response["main_answer"],
            "notes":response['additional_notes']
        })
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": F"Internal server error: {str(e)}"}), 500
    
# Flask route for uploading new datasets (Admin feature)
@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    print(f"Incoming {request.method} request to {request.path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Payload: {request.get_data(as_text=True)}")

    """ Admin endpoint for uploading new herbal datasets. """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        data = json.load(file)
        vectors = []

        for i, entry in enumerate(data):
            # Clean and prepare text
            raw_text = entry.get("content", "") or entry.get("description", "") or entry.get("remedy_description", "")
            cleaned_text = clean_output(raw_text)

            # Embed text
            vector = hf_embeddings.embed_query(cleaned_text)

            # Attach metadata
            metadata = clean_metadata(entry)
            vectors.append({
                "id": f"upload-{i}",
                "values": vector,
                "metadata": metadata
            })

        # Upload to Pinecone
        safe_pinecone_upsert(index, vectors)
        return jsonify({"message": f"Uploaded {len(vectors)} entries to the vector database."})

    except Exception as e:
        logging.error(f"Error uploading dataset: {str(e)}")
        return jsonify({"error": "Error uploading dataset."}), 500

@app.route("/health", methods=["GET"])
def health_check():
    try:
        index.describe_index_stats()
        return jsonify({"pinecone": "healthy"}), 200
    except Exception as e:
        print("Pinecone health check failed:", str(e))  # optional logging
        return jsonify({
            "pinecone": "unhealthy",
            "error": str(e)
        }), 200  # <-- Force 200 so Render doesn’t fail deployment

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
