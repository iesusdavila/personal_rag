import llama_cpp
from qdrant_client import QdrantClient
from rich.console import Console

embedding_llm = llama_cpp.Llama(
  model_path="models/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/db9d1fe0f31addb4978201b2bf3e577f3f8900d2/gguf/mxbai-embed-large-v1-f16.gguf", 
  embedding=True,
  verbose=False
)

client = QdrantClient(path="embeddings")


# Search
search_query = "What is the Ieus Davila email?"
query_vector = embedding_llm.create_embedding(search_query)['data'][0]['embedding']
search_result = client.search(
  collection_name="podcast",
  query_vector=query_vector,
  limit=5
)

print(search_result)

template = """You are a helpful assistant who answers questions using only the provided context.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

llm = llama_cpp.Llama(
  model_path="models/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf",
  verbose=False
)

stream = llm.create_chat_completion(
    messages = [
      {"role": "user", "content": template.format(
        context = "\n\n".join([row.payload['text'] for row in search_result]),
        question = search_query      
      )}
    ],
    stream=True
)

for chunk in stream:
  print(chunk['choices'][0]['delta'].get('content', ''), end='')