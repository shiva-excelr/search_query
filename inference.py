import requests



def generate_embeddings():
    # return self.ollama_embedding_function(texts)
    url = "http://localhost:11434/api/embed"
    payload = {"model": 'nomic-embed-text', "input": ["Why is the sky blue?", "Why is the grass green?"]}
    response = requests.post(url, json=payload)
    print("response", response.text)
    if response.status_code == 200:
        return response.json().get("embedding", [])
    else:
        raise Exception(f"Failed to get embeddings: {response.text}")



print(generate_embeddings())