import traceback

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from qdrant_vector_store import QdrantVectorStore
from typing import List, Dict, Union
import asyncio
app = FastAPI()

vector_store =QdrantVectorStore(collection_name='marketplace',query="SELECT * FROM marketplace")

class VectorRequest(BaseModel):
    sql_data: Dict
    unique_id: str

class SearchResult(BaseModel):
    id: int
    score: float
    payload: Dict

class ResponseModel(BaseModel):
    status: str = "success"
    statuscode: int = 200
    result: List[SearchResult] = None

@app.post("/upload/", response_model=List[dict])
async def fetch_all_transactions(data: Union[VectorRequest, List[VectorRequest]], background_tasks: BackgroundTasks,pre_process: bool = False):
   try:
       id = data.get("guid")

       if isinstance(data, VectorRequest):
           records = [data]
       elif isinstance(data, list) and all(isinstance(item, VectorRequest) for item in data):
           records = data
       else:
           raise HTTPException(
               status_code=400, detail="Request must be a single object or a list of VectorRequest objects."
           )
       background_tasks.add_task(vector_store.add_vectors, records,pre_process)
       return {"message": f"Vector upload task for ID {id} added to background"}

   except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding vector stores: {str(e)}")


@app.get("/search/", response_model=ResponseModel)
async def search(query: str = Query(..., description="Search query text"),
    top_k: int = Query(10, description="Number of top results to return", ge=1, le=50),
):
    try:
        results = vector_store.search_vector_v2(query, top_k)

        response = [
            SearchResult(id=result.get("id"), score=result.get("score"), payload=result)
            for result in results
        ]

        if results:
            return ResponseModel(result = response)

        else:
            return JSONResponse(content="No Records found for the query",status_code=200)

    except Exception as e:
        print(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

