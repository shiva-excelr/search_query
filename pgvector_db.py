from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from database import *


connection = "postgresql+psycopg://postgres:test@localhost:5432/vector_db"  # Uses psycopg3!
collection_name = "test"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)


docs = [
   ]

data = get_all_transactions()
ids = []
for document in data:
    ids.append(document["guid"])
    response = "Response "+document.get("response") if document.get("response") else ''
    request = "\n"+" Request " +document.get("request") if document.get("request") else ''
    text = response + request

    docs.append(Document(
        page_content= text,
        metadata={**{key: value for key, value in document.items() if key not in ["response", "request",'date']}, "text":text,'date':str(document.get("date"))},
    ))
# vector_store.add_documents(documents=docs,ids=ids)

results = vector_store.similarity_search(
    "ref id is 805414040578", k=10
)
for doc in results:
    print("GUID", doc.metadata.get("guid"))
    print(f"* {doc.page_content}")