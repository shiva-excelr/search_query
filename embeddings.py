from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from openai import OpenAI
import os

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


import chromadb.utils.embedding_functions as embedding_functions

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)


def generate_embedding(text: list[str]):

    try:
        vector = model.encode(text)
        vector = vector.tolist()

        return vector
    except:
        return None



def ranked_scores(query_embedding,vector_embeddings, texts, query):
    similarity_scores = cos_sim(query_embedding, vector_embeddings)
    sorted_indices = similarity_scores.argsort(descending=True)
    ranked_scores = similarity_scores[sorted_indices]
    rearrange_texts = []
    for index in sorted_indices:
        rearrange_texts.append(texts[index])


    return rearrange_texts

class OpenAIEmbeddings:
    def __init__(self):
        self.model =  "text-embedding-3-small"
        self. client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_embeddings(self,texts):
        """
        Generate embeddings for a list of texts using OpenAI's embedding model.

        :param texts: List of strings to generate embeddings for
        :return: List of embedding vectors
        """
        # Use text-embedding-3-small for a balance of performance and cost
        successful_embeddings = []
        successful_texts = []
        for index,text in enumerate(tqdm(texts)):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model)

                # Extract the embedding vectors
                # embeddings = [embed.embedding for embed in response.data]
                embeddings = response.data[0].embedding
                successful_embeddings.append(embeddings)
                successful_texts.append(text)
                print("index", index)

            except Exception as e:
                print(text)
                print(f"An error occurred: {e}")
                print("\n")
                continue
        return successful_embeddings, successful_texts

