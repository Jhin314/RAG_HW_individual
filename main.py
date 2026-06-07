import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

#Ingestion code to upload vectors to the database.


# Configuration
CSV_PATH = "medium-english-50mb.csv"
INDEX_NAME = "rag-index02"
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
TESTING_MODE = False # Start with a small subset to save budget

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize LLMod.ai Client
# This uses the standard OpenAI library but redirects the traffic to LLMod
client = OpenAI(
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL")
)

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    # Corrected to access the first element of the data array
    return response.data[0]

def ingest_data():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    if TESTING_MODE:
        df = df.head(100) # Subset for validation

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=102, # 20% overlap
        length_function=len
    )

    index = pc.Index(INDEX_NAME)

    batch_vectors = []

    print("Chunking and embedding...")
    for idx, row in df.iterrows():
        chunks = text_splitter.split_text(str(row['text']))

        for i, chunk in enumerate(chunks):
            chunk_id = f"article_{idx}_chunk_{i}"
            embedding = get_embedding(chunk).embedding

            # Storing metadata for retrieval
            metadata = {
                "article_id": str(idx),
                "title": str(row['title']),
                "authors": str(row['authors']),
                "chunk": chunk
            }

            batch_vectors.append(
                {"id": chunk_id, "values": embedding, "metadata": metadata}
            )

            if len(batch_vectors) >= 50:
                index.upsert(vectors=batch_vectors)
                batch_vectors = []
                print("Upserted 50 chunks...")

    if batch_vectors:
        index.upsert(vectors=batch_vectors)
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()