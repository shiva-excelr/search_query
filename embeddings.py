from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


import chromadb.utils.embedding_functions as embedding_functions

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="all-minilm",
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

