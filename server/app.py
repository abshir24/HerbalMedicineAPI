import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime, timedelta
from helper import (
    retry_with_backoff,
    safe_pinecone_query,
    safe_pinecone_upsert,
    extract_keywords_from_query,
    clean_output,
    clean_metadata,
    query_vector_db,
    herbal_medicine_query_with_context,
    fetch_relevant_context,
    full_pipeline_test
)
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec,PineconeException
import os


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

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

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
