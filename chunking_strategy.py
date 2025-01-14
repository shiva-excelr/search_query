import re
import uuid
from itertools import islice

from langchain_text_splitters import CharacterTextSplitter
import tiktoken
from embeddings import OpenAIEmbeddings
CHUNK_LENGTH = 1000
OVERLAP = 100

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

embeddings = OpenAIEmbeddings()

def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be greater than 0")
    it = iter(iterable)
    while batch:= tuple(islice(it,n)):
        yield batch

def contains_alphanumeric(text):
    return bool(re.search(r'\w',text,re.UNICODE))

def chunk_text_with_overlap(text, max_tokens=CHUNK_LENGTH, overlap=OVERLAP):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)

        if len(chunk) >= max_tokens:
            chunks.append(' '.join(chunk))
            chunk = chunk[-overlap:]

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks



def chunked_tokens(text, encoding = encoding, chunk_size = CHUNK_LENGTH):

    tokens = encoding.encode(text)
    chunks = batched(tokens, chunk_size)
    yield from chunks





def get_chunk_tokens_and_embeds(text):
    chunk_embeddings = []
    chunk_lens = []
    result = []

    for chunked_token in chunked_tokens(text, encoding=encoding):
        chunked_token_text = encoding.decode(chunked_token)
        if contains_alphanumeric(chunked_token_text):
            embedding =  embeddings.generate_embeddings(chunked_token_text)
            chunk_embeddings.append(embedding)
            chunk_lens.append(len(chunked_token))

            result.append({
                'text': chunked_token_text,
                'tokens': chunked_token,
                'length': len(chunked_token),
                'embeddings': embedding
            })


    return result



def get_chunks_and_embeds(data):

    chunked_docs = []
    doc_id = str(uuid.uuid4())  #hashkey on text

    for doc in data:
        chunk_id = str(uuid.uuid4())
        for index,dense_chunk in enumerate(get_chunk_tokens_and_embeds(doc['text']), 1):
            chunk_text = dense_chunk['text']
            vector = dense_chunk['embeddings']

            chunked_docs.append({"text":chunk_text,'embeddings':vector,'chunkId':chunk_id,'chunkPosition':index,})


