import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import pandas as pd

from database import  get_all_transactions


def process_row(row):
    """
    Generate a clean and meaningful context string for each API request based on the row's data.
    """
    context = (
        f"The API request belongs to the collection '{row['Type']}' and is named '{row['Purpose']}'. "
        f"It processes a '{row['Protocol']}' request with the content type '{row['DataFormat']}'. "
        f"The request is identified by packager ID '{row['GUID']}' and is associated with team ID '{row['HeaderSize']}'. "
        f"Settings for this request are specified as '{row['Metadata']}'. "
        f"The request body is: {row['Payload']}. "
        f"The response for the request is: {row['Response']}. "
        f"The description of the request is: {row['Description']}."
    )
    return context

data = get_all_transactions(query="SELECT * FROM marketplace")
# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Initialize embedding model and Qdrant client
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(url="http://localhost:6333")

# Step 3: Process rows and generate context strings
points = []
for _, row in df.iterrows():
    context = process_row(row)
    embedding = embedding_model.encode(context).tolist()
    metadata = {
        "ID": row["ID"],
        "GUID": row["GUID"],
        "Type": row["Type"],
        "Purpose": row["Purpose"],
        "Protocol": row["Protocol"],
        "DataFormat": row["DataFormat"],
        "HeaderSize": row["HeaderSize"],
        "Description": row["Description"],
        "Payload": row["Payload"],
        "Response": row["Response"],
        "Text": context,
    }
    point = PointStruct(id=row["ID"], vector=embedding, payload=metadata)
    points.append(point)

# Step 4: Upload points to Qdrant
collection_name = "visa_requests"
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config={"size": len(embedding), "distance": "Cosine"},
)
qdrant_client.upsert(collection_name=collection_name, points=points)

# Step 5: Test vector search
def test_query(query):
    """Test a query by searching in Qdrant."""
    query_vector = embedding_model.encode(query).tolist()
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
    )
    for result in results:
        print("Match:", result.payload["Text"])
        print("Metadata:", result.payload)

# Example test
user_query = "create a VISA request for auth"
test_query(user_query)
