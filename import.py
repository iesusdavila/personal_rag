import uuid

import jsonlines
import time
import llama_cpp

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from itertools import islice
from langchain_core.documents import Document
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import chunk
import fitz

file = "docs/CV_ES.pdf"

def read_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

text_content = read_pdf(file)

print(f"PDF content length: {len(text_content)} characters")
print(f"First 200 characters: {text_content[:200]}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# Opción 1: Pasar una lista con el texto completo
documents = text_splitter.create_documents([text_content])

# Opción 2 (alternativa): Usar split_text directamente
# chunks = text_splitter.split_text(text_content)
# documents = [Document(page_content=chunk) for chunk in chunks]

print(f"Number of document chunks: {len(documents)}")
print(f"First chunk preview: {documents[0].page_content[:100]}...")

llm = llama_cpp.Llama(
  model_path="models/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/db9d1fe0f31addb4978201b2bf3e577f3f8900d2/gguf/mxbai-embed-large-v1-f16.gguf", 
  embedding=True, 
  verbose=False
)

batch_size = 100
documents_embeddings = []
batches = list(chunk(documents, batch_size))
start = time.time()
for i, batch in enumerate(batches):
    print(f"Processing batch {i+1}/{len(batches)}")
    embeddings = llm.create_embedding([item.page_content for item in batch])
    documents_embeddings.extend(
        [ 
            (document, embeddings['embedding'])  
            for document, embeddings in zip(batch, embeddings['data'])
        ]
    )
end = time.time()
char_per_second = len(''.join([item.page_content for item in documents])) / (end-start)
print(f"Time taken: {end-start:.2f} seconds / {char_per_second:,.2f} characters per second")

# Init client and create collection
client = QdrantClient(path="embeddings")

# Delete collection if it exists (for clean restart)
try:
    client.delete_collection(collection_name="podcast")
    print("Deleted existing collection")
except:
    print("No existing collection to delete")

client.create_collection(
    collection_name="podcast",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# Store documents
points = [
    PointStruct(
        id = str(uuid.uuid4()),
        vector = embeddings,
        payload = {
            "text": doc.page_content
        }
    )
    for doc, embeddings in documents_embeddings
]

operation_info = client.upsert(
    collection_name="podcast",
    wait=True,
    points=points
)

print(f"Stored {len(points)} document chunks in Qdrant")

search_query = "What is the Ieus Davila email?"
query_vector = llm.create_embedding(search_query)['data'][0]['embedding']
search_result = client.search(
    collection_name="podcast",
    query_vector=query_vector,
    limit=2,
)

print(f"\nSearch results for query '{search_query}':")
print(search_result)
# for i, result in enumerate(search_result):
#     print(f"\nResult {i+1}:")
#     print(f"ID: {result.id}")
#     print(f"Score: {result.score:.4f}")
#     print(f"Text preview: {result.payload['text']}...")
#     print(f"Full text length: {len(result.payload['text'])} characters")