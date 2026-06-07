import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# 1. Initialize Pinecone Client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Updated to point to your new correctly dimensioned index!
index = pc.Index("rag-index03")

# 2. Initialize LLMod.ai Client
#client = OpenAI(
#    api_key=os.environ.get("LLMOD_API_KEY"),
#    base_url=os.environ.get("LLMOD_BASE_URL"),
#    timeout=60 #TODO correct ?
#)

# Models & Config
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
CHUNK_SIZE = 320
OVERLAP_RATIO = 0.2
TOP_K = 15

# Exact required system prompt
SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data."
Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful."""

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    """Health check so the deployment root confirms the function is live."""
    return {"status": "ok", "service": "RAG Medium-article assistant"}

@app.get("/api/stats")
def get_stats():
    """Returns the exact configuration chosen for the RAG system."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

@app.post("/api/prompt")
def query_rag(request: QueryRequest):
    question = request.question

    print("[DEBUG] index.py: Received request")

    client = OpenAI(
        api_key=os.environ["LLMOD_API_KEY"],
        base_url=os.environ["LLMOD_BASE_URL"]
    )

    # 1. Embed the query
    embed_response = client.embeddings.create(
        input=question,
        model=EMBEDDING_MODEL
    )

    print("[DEBUG] index.py: Embedding done")

    # Using.embedding to safely unpack the list, just like in the ingestion script
    query_vector = embed_response.data[0].embedding

    # 2. Retrieve from Pinecone
    search_results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )

    print("[DEBUG] index.py: Pinecone query done")

    # 3. Format Context
    context_chunks = []
    formatted_context_text = ""

    for match in search_results['matches']:
        meta = match.get('metadata', {})
        context_chunks.append({
            "article_id": meta.get("article_id", "unknown"),
            "title": meta.get("title", "Untitled"),
            "chunk": meta.get("chunk", ""),
            "score": match.get('score', 0.0)
        })
        formatted_context_text += f"Title: {meta.get('title')}\nAuthor: {meta.get('authors')}\nPassage: {meta.get('chunk')}\n\n"

    # 4. Construct User Prompt
    user_prompt = f"Context:\n{formatted_context_text}\n\nUser Question: {question}"

    # 5. Generate Response via Chat Model
    chat_response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    print("[DEBUG] index.py: Chat done")

    final_answer = chat_response.choices[0].message.content

    del client

    # 6. Return strictly formatted JSON
    return {
        "response": final_answer,
        "context": context_chunks,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt
        }
    }