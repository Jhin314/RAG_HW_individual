import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()


#Better version of the ingestion code in the now removed main.py

# Configuration
CSV_PATH = "medium-english-50mb.csv"
INDEX_NAME = "rag-index03" #Name of latest index used (Developer's note: indexes 1 and 2 were used for debugging before scaling)
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
TESTING_MODE = False # Start with a small subset to save budget. Since we don't need to do so anymore we set to False.

CHUNK_BATCH_UPLOAD_SIZE = 50
#Number of chunks to collect before processing with APIs.
#Developer's note: I started small for debugging purposes, and then scaled as needed.
#The higher it is, the less requests we need to make to the APIs overall.


# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize LLMod.ai Client
client = OpenAI(
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL")
)

def get_embeddings_batch(texts):
    """Takes a list of texts and returns a list of embeddings all at once."""
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    # The API returns a list of objects, we extract the .embedding from each one
    return [item.embedding for item in response.data]

def ingest_data():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    if TESTING_MODE: #Used for debugging and scaling
        df = df.head(100)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=102, #i.e overlap ratio is 0.2 since 512 * 0.2 = 102
        length_function=len
    )

    index = pc.Index(INDEX_NAME)

    # We will temporarily store the raw data here before embedding
    text_batch = []
    meta_batch = []
    id_batch = []

    print("Chunking and batch embedding...")
    for idx, row in df.iterrows():
        chunks = text_splitter.split_text(str(row['text']))

        for i, chunk in enumerate(chunks):
            # Collect the raw text, ID, and metadata
            id_batch.append(f"article_{idx}_chunk_{i}")
            text_batch.append(chunk)
            meta_batch.append({
                "article_id": str(idx),
                "title": str(row['title']),
                "authors": str(row['authors']),
                "chunk": chunk
            })

            # When we have collected batch items, process them all at once!
            if len(text_batch) >= CHUNK_BATCH_UPLOAD_SIZE:
                print(f"Sending {CHUNK_BATCH_UPLOAD_SIZE} chunks to the embedding model...")
                embeddings = get_embeddings_batch(text_batch)

                # Zip them together into the format Pinecone needs
                pinecone_vectors = [
                    {"id": chunk_id, "values": emb, "metadata": meta}
                    for chunk_id, emb, meta in zip(id_batch, embeddings, meta_batch)
                ]

                index.upsert(vectors=pinecone_vectors)
                print(f"Upserted {CHUNK_BATCH_UPLOAD_SIZE} chunks into Pinecone.\n")

                # Clear the lists for the next batch
                text_batch.clear()
                meta_batch.clear()
                id_batch.clear()

    # Catch any leftover chunks that didn't make a full set of 50 at the end
    if text_batch:
        print(f"Sending final {len(text_batch)} chunks...")
        embeddings = get_embeddings_batch(text_batch)
        pinecone_vectors = [
            {"id": chunk_id, "values": emb, "metadata": meta}
            for chunk_id, emb, meta in zip(id_batch, embeddings, meta_batch)
        ]
        index.upsert(vectors=pinecone_vectors)

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()