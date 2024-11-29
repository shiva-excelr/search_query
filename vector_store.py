import copy
import json
import re
import traceback
import uuid
import xml.etree.ElementTree as ET
from enum import Enum

from bs4 import BeautifulSoup
from datetime import datetime

from qdrant_client import QdrantClient

import chromadb
from embeddings import generate_embedding
from base_wrapper import  *
from embeddings import ollama_ef

from qdrant_client.models import Distance, VectorParams
from database import get_all_transactions
from qdrant_client.models import PointStruct, MultiVectorConfig,MultiVectorComparator
from utils import *

class Include(Enum):
    EMBEDDING = "embeddings"
    DOCUMENTS = "documents"
    METADATA = "metadata"


@singleton
class VectorStore:
    def __init__(self):
         self.client = chromadb.PersistentClient("/Users/mac/PycharmProjects/PythonProject/chromadb/")

    def formatted_json_str(self,json_str,default=False):
        try:
            if json_str and 'error' not in json_str.lower():
                data = json.loads(json_str)

                relevant_fields = []
                for key, value in data.items():
                    if isinstance(value, str):
                        relevant_fields.append(key + ' ' + value)
                    elif isinstance(value, (list, dict)):
                        relevant_fields.append(key + ' ' + json.dumps(value))

                combined_text = " ".join(relevant_fields)

                normalized_text = " ".join(combined_text.split())

                return " " + normalized_text

        except json.JSONDecodeError:
            print("Invalid JSON string")
            return json_str if default else None

    def formatted_xml_str(self,xml_data):
        soup = BeautifulSoup(xml_data, "xml")

        data = soup.contents[0]

        def str_elements(data):
            st = '' + data.name
            for t in data.contents:
                if not isinstance(t, str):
                    att = t.attrs
                    st += self.formatted_json_str(json.dumps(att),default=True)
                    if t.contents:
                        st += " " + str_elements(t)

            return st + ' '

        return str_elements(data)

    def preprocess_xml(self,xml_str):
        # lines = xml_str.strip().splitlines()
        # if lines[0].startswith('<?xml'):
        #     lines = lines[1:]
        #
        # wrapped_xml = "<Root>" + "".join(lines) + "</Root>"
        return xml_str

    def convert_xml_to_json(self,xml_str):
        try:
            preprocessed_xml = self.preprocess_xml(xml_str)
            root = ET.fromstring(preprocessed_xml)
            result = {child.tag: child.attrib for child in root}
            return json.dumps(result, indent=4)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return self.formatted_xml_str(xml_str)

    def convert_payload(self,payload):
        try:
            if isinstance(payload, str) and payload.strip().startswith(("<xml", "<?xml")):
                preprocessed_xml = self.preprocess_xml(payload)
                formatted_xml = self.formatted_xml_str(preprocessed_xml)
                return None
            elif isinstance(payload, str) and payload.strip().startswith(("<iso")):
                soup = BeautifulSoup(payload, 'xml')
                values = [field['value'] for field in soup.find_all('field')]
                text = '\n'.join(values)
                return None
            elif bool(re.compile(r'<[^>]+>').search(payload)):
                soup = BeautifulSoup(payload, 'html.parser')
                html_str = soup.get_text(strip=True, separator='\n')
                html_str = "\n".join(dict.fromkeys(html_str.splitlines()))
                return None

            return self.formatted_json_str(payload)

        except Exception as e:
            return self.formatted_json_str(payload)

    def chunk_text_with_overlap(self,text, max_tokens=450, overlap=50):

        words = text.split()
        chunks = []
        chunk = []

        for word in words:
            chunk.append(word)

            if len(chunk) >= max_tokens:
                chunks.append(' '.join(chunk))  # Join words to form the chunk
                chunk = chunk[-overlap:]

        # Add the final chunk if there's any leftover text
        if chunk:
            chunks.append(' '.join(chunk))

        return chunks

    def store_vectors(self,db_data):

        collection = self.client.get_or_create_collection("transactions")

        for transaction in db_data:
            act_response = self.convert_payload(transaction.get('response'))
            payload = self.convert_payload(transaction.get('request'))


            act_response_vector = generate_embedding(act_response)
            payload_vector = generate_embedding(payload)



            if act_response_vector:
                collection.add(
                    documents=[act_response],
                    embeddings=[act_response_vector],
                    metadatas=[{"response":True,"text":act_response,"date": str(transaction.get("date")), "guid": transaction.get("guid")}],
                    ids=[transaction.get("guid")]
                )

            if payload_vector:
                collection.add(
                    documents=[payload],
                    embeddings=[payload_vector],
                    metadatas=[{"request":True,"text":payload,"date": str(transaction.get("date")), "guid": transaction.get("guid")}],
                    ids=[transaction.get("guid")]
                )



    def store_vectors2(self,db_data):

        collection = self.client.get_or_create_collection("transactions", metadata={"embedding_dimension": 768})

        documents = []
        documents1 = []
        metadatas =[]
        metadatas1=[]
        ids = []
        ids1=[]
        act_responses = []
        payload_responses = []



        for transaction in db_data:
            try:
                print(transaction.get("guid"))

                if transaction.get("guid") == "2886":
                    pass


                act_response_vector =generate_embedding([transaction.get('response') if transaction.get('response') else None])
                payload_vector = generate_embedding([transaction.get('request')  if transaction.get('request') else None])

                act_response = self.convert_payload(transaction.get('response')if transaction.get('response') else None)
                payload = self.convert_payload(transaction.get('request') if transaction.get('request') else None)




                if act_response and act_response_vector:
                   chunks = self.chunk_text_with_overlap(act_response)
                   for id,chunk in enumerate(chunks,1):
                       act_responses.append(chunk.lower())
                       documents.append(chunk.lower())
                       metadatas.append({"chunk_id":str(transaction.get("guid")) + "@ "+'response'+str(id),"response": True, "text": act_response, "date": str(transaction.get("date")),
                                         "guid": transaction.get("guid")})
                       ids.append(str(transaction.get("guid")) + "@ "+'response'+str(id))


                if payload and payload_vector:
                   chunks_1 = self.chunk_text_with_overlap(payload)
                   for id, chunk in enumerate(chunks_1, 1):


                       payload_responses.append(chunk.lower())
                       documents1.append(chunk.lower())
                       metadatas1.append({"chunk_id":str(transaction.get("guid")) + "@ "+'payload'+str(id),"request": True, "text": payload, "date": str(transaction.get("date")),
                                          "guid": transaction.get("guid")})
                       ids1.append(str(transaction.get("guid")) +'@ '+'payload'+str(id))

            except Exception as e:
                print(e)
                print(traceback.format_exc())
                return










        act_response_vectors = ollama_ef(act_responses)
        payload_vectors = ollama_ef(payload_responses)







        collection.add(
            documents=documents,
            embeddings=act_response_vectors,
            metadatas=metadatas,
            ids=ids
        )
        collection.add(
            documents=documents1,
            embeddings=payload_vectors,
            metadatas=metadatas1,
            ids=ids1
        )


    def search_vectors(self,query: str, top_k: int = 20):
        query_vector = ollama_ef([query])


        collection = self.client.get_collection("transactions")
        results = collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["embeddings", "metadatas",'documents','distances']


        )
        metadata_list = results['metadatas'][0]
        document_list = results['documents'][0]
        id_list = results['ids'][0]

        # Create a combined list of tuples for sorting
        combined = list(zip(metadata_list, document_list, id_list))

        # Sort the combined list based on the 'date' field in metadata
        sorted_combined = sorted(
            combined,
            key=lambda x: datetime.strptime(x[0]['date'], "%Y-%m-%d %H:%M:%S.%f")
        )

        # Unpack the sorted data back into separate lists
        sorted_metadatas, sorted_documents, sorted_ids = zip(*sorted_combined)

        # Update the results object
        results['metadatas'][0] = list(sorted_metadatas)
        results['documents'][0] = list(sorted_documents)
        results['ids'][0] = list(sorted_ids)

        top_result =[result for result in results['ids'][0]]
        print(" ".join([result for result in results['documents'][0]]))



        return top_result



class QdrantVectorStore:

    def __init__(self,embedding_dimension=384):

        self.collection_name = "search_analytics"
        self.client = QdrantClient(url="http://localhost:6333")

        self.vector_store = VectorStore()

        self.embeddings=ollama_ef

        if not self.client.collection_exists(collection_name=self.collection_name):
            self.collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE, multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                )),

            )
        else:
            self.collection = self.client.get_collection(collection_name=self.collection_name)
        self.data = get_all_transactions()



    def add_vectors(self):
        # data_copy = copy.deepcopy(self.data)
        for idx,transaction in enumerate(self.data):
            try:
                act_response_vector =generate_embedding([transaction.get('response') if transaction.get('response') else None])
                payload_vector = generate_embedding([transaction.get('request')  if transaction.get('request') else None])

                # if transaction.get("guid") in ["bc8a2ff4-7ac9-4a1a-bb16-f7389059e3a6","2810", "2707","2691","2708"]:
                #      pass

                act_response = self.vector_store.convert_payload(transaction.get('response')if transaction.get('response') else None)
                payload = self.vector_store.convert_payload(transaction.get('request') if transaction.get('request') else None)





                if act_response and act_response_vector:
                   # chunks = self.vector_store.chunk_text_with_overlap(act_response)
                   transaction["response"] = act_response.lower()
                   transaction["documents"] = act_response.lower()
                   transaction["metadata"] = {"response": True, "text": act_response, "date": str(transaction.get("date")),
                                     "guid": transaction.get("guid")}
                   transaction["responseId"] = generate_hash_key(str(transaction.get("guid")) + "@"+'response')
                   # transaction["responseVector"] =



                if payload and payload_vector:
                   # chunks_1 = self.vector_store.chunk_text_with_overlap(payload)
                   transaction["request"] = payload.lower()
                   transaction["documents1"] = payload.lower()
                   transaction["metadata1"] = {
                                              "request": True, "text": payload,
                                              "date": str(transaction.get("date")),
                                              "guid": transaction.get("guid")}
                   transaction["requestId"] = generate_hash_key(str(transaction.get("guid")) + "@" + 'request')
                   # transaction["requestVector"] =


                if not payload and not act_response:
                    self.data.remove(transaction)
                    continue


            except Exception as e:
                print(e)
                print(traceback.format_exc())
                return

        points =[]

        for idx,doc in enumerate(self.data):
            try:
                if doc.get("response") and doc.get("responseId"):
                    embed = ollama_ef(doc.get("response"))
                    pay= {k: v for k, v in doc.items() if
                                         k not in ["requestId", "responseVector", "metadata1", "documents1", "request"]}
                    points.append(PointStruct(id=doc.get("responseId"), vector=embed,
                                payload=pay))
            except Exception as e:
                print(e)


        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=doc.get("responseId"), vector=ollama_ef(doc.get("response")), payload={k: v for k, v in doc.items() if k not in ["requestId","responseVector","metadata1","documents1","request"]}) for doc in self.data if doc.get("response") and doc.get("responseId")
            ],
        )

        operation_info1 = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=doc.get("requestId"), vector=ollama_ef(doc.get("request")), payload={k: v for k, v in doc.items() if k not in ["responseVector","responseId","metadata","documents","response"]}) for doc in self.data if doc.get("request") and doc.get("requestId")
            ],
        )

    def search_vector(self,query, top_k =10):
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=ollama_ef([query]),
            limit=top_k,
        ).points

        for hit in hits:
            print(hit.payload.get("guid"), "score:", hit.score,'\n',"text:",hit.payload.get('documents',hit.payload.get('documents1')))




if __name__ == "__main__":
    # vector_store = VectorStore()
    # db_data = get_all_transactions()
    # vector_store.store_vectors2(db_data)
    # collections = vector_store.client.list_collections()
    # print(vector_store.client.get_collection("transactions").metadata)


    # print(vector_store.search_vectors("Sarjapur Road"))
    # print(vector_store.search_vectors("refId is 804813039157"))
    # print(vector_store.search_vectors("AXIS0000058"))

    qdrant = QdrantVectorStore()
    # qdrant.add_vectors()
    qdrant.search_vector("AXIS0000058")






