

from fastapi import FastAPI, HTTPException
from database import  get_all_transactions
from vector_store import VectorStore
from typing import List


app = FastAPI()



@app.get("/transactions", response_model=List[dict])
def fetch_all_transactions():
   try:
        transactions = get_all_transactions()

        VectorStore().store_vectors(transactions)

        return {"message": "Transactions added and vectors updated successfully!"}
   except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding vector stores: {str(e)}")


@app.get("/search/{query}", response_model=List[str])
def search(query: str, top_k: int = 5):
    return VectorStore().search_vectors(query, top_k)

