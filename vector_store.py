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
from tqdm import tqdm

import chromadb
from embeddings import generate_embedding, OpenAIEmbeddings
from base_wrapper import  *
from embeddings import ollama_ef

from qdrant_client.models import Distance, VectorParams, BinaryQuantization, BinaryQuantizationConfig
from database import get_all_transactions
from qdrant_client.models import PointStruct, MultiVectorConfig,MultiVectorComparator
from utils import *


open_ai_embeds = OpenAIEmbeddings()
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
            if json_str:
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
                return formatted_xml
            elif isinstance(payload, str) and payload.strip().startswith(("<iso")):
                soup = BeautifulSoup(payload, 'xml')
                values = [field['value'] for field in soup.find_all('field')]
                text = '\n'.join(values)
                return text
            elif bool(re.compile(r'<[^>]+>').search(payload)):
                soup = BeautifulSoup(payload, 'html.parser')
                html_str = soup.get_text(strip=True, separator='\n')
                html_str = "\n".join(dict.fromkeys(html_str.splitlines()))
                return html_str

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

    def __init__(self,collection_name = "openaisearch",embedding_dimension=1536):

        self.collection_name = collection_name

        self.client = QdrantClient(url="http://localhost:6333", timeout=3000)

        self.vector_store = VectorStore()

        self.embeddings=ollama_ef


        if not self.client.collection_exists(collection_name=self.collection_name):
            # self.collection = self.client.create_collection(
            #     collection_name=self.collection_name,
            #     vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE, multivector_config=MultiVectorConfig(
            #         comparator=MultiVectorComparator.MAX_SIM
            #     )),
            #
            # )
            self.collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(
                        always_ram=True,
                    ),
                ),
            )
        else:
            self.collection = self.client.get_collection(collection_name=self.collection_name)

        # self.client.delete_collection(self.collection_name)


        self.data = get_all_transactions()


    def delete_collection(self):
        self.client.delete_collection(self.collection_name)

    def add_vectors(self):
        data_copy = copy.deepcopy(self.data)
        vec = []
        guids = []
        payloads =[]
        for idx,transaction in enumerate(tqdm(data_copy)):
            try:


                # if transaction.get("guid") in ["bc8a2ff4-7ac9-4a1a-bb16-f7389059e3a6","2810", "2707","2691","2708"]:
                #      pass
                #
                if transaction.get("guid") in ["01c92f92-02cb-4ea2-b3f4-005ef0584a7f"]:
                     pass



                # act_response = self.vector_store.convert_payload(transaction.get('response')if transaction.get('response') else None)

                act_response = transaction.get("response")







                if act_response:
                   vec.append(act_response.lower())
                   # embed = open_ai_embeds.generate_embeddings([act_response])
                   # embeds.append({transaction.get("guid"):embed})
                   # transaction["embeddings"] = embeds

                   # transaction["response"] = act_response.lower()
                   # transaction["documents"] = act_response.lower()
                   # transaction["metadata"] = {"response": True, "text": act_response, "date": str(transaction.get("date")),
                   #                   "guid": transaction.get("guid")}
                   # transaction["responseId"] = generate_hash_key(str(transaction.get("guid")) + "@"+'response')

                   guids.append(transaction["guid"])
                   payloads.append({"text":act_response.lower(),"response": True, "text": act_response.lower(), "date": str(transaction.get("date")),"guid": transaction.get("guid")})
                else:
                    data_copy.remove(transaction)




            except Exception as e:
                print(e)
                print(traceback.format_exc())
                return



        # for idx,doc in enumerate(data_copy):
        #     try:
        #         if doc.get("response") and doc.get("responseId"):
        #             embed = ollama_ef(doc.get("response"))
        #             pay= {k: v for k, v in doc.items() if
        #                                  k not in ["requestId", "responseVector", "metadata1", "documents1", "request"]}
        #             points.append(PointStruct(id=doc.get("responseId"), vector=embed,
        #                         payload=pay))
        #     except Exception as e:
        #         print(e)

        #


        # embeddings, texts = open_ai_embeds.generate_embeddings([i['text'] for i in payloads])

        # with open("embeds_without_preprocess.json",'w') as f:
        #     json.dump(embeddings,f)
        #
        # with open("texts_without_preprocess.json", 'w') as f:
        #     json.dump(texts, f)

        # with open('texts.json', 'r') as f:
        #     texts = json.load(f)
        #
        # with open('embeds.json', 'r') as f:
        #     embeddings = json.load(f)


        with open('texts_without_preprocess.json', 'r') as f:
            texts = json.load(f)

        with open('embeds_without_preprocess.json', 'r') as f:
            embeddings = json.load(f)


        # operation_info = self.client.upsert(
        #     collection_name=self.collection_name,
        #     wait=True,
        #     points=[
        #         PointStruct(id=doc.get("responseId"), vector=v, payload={k: v for k, v in doc.items() if k not in ["requestId","responseVector","metadata1","documents1","request"]}) for v,doc in zip(embeddings, data_copy) if doc.get("response") and doc.get("responseId")
        #     ],
        # )

        points = [
                PointStruct(id=id, vector=v,
                            payload=doc)
                for id,(v, doc) in enumerate(tqdm(zip(embeddings, payloads)))
            ]

        operation_info = self.client.upload_points(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )




        # operation_info1 = self.client.upsert(
        #     collection_name=self.collection_name,
        #     wait=True,
        #     points=[
        #         PointStruct(id=doc.get("requestId"), vector=ollama_ef(doc.get("request")), payload={k: v for k, v in doc.items() if k not in ["responseVector","responseId","metadata","documents","response"]}) for doc in self.data if doc.get("request") and doc.get("requestId")
        #     ],
        # )

    def search_vector(self,query, top_k =10):
        query_embed,text =open_ai_embeds.generate_embeddings([query])
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embed[0],
            limit=top_k,
        ).points

        # results = []
        # for i in hits:
        #     i.payload["score"] = i.score
        #     results.append(i.payload)

        results = [{**i.payload, "score": i.score} for i in hits]

        sorted_results = sorted(results, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S.%f'),reverse=True)

        for hit in sorted_results:
            print(hit.get("guid"),"score:",hit.get('score'),"date:", hit.get('date'),'\n',"text:",hit.get('text'))
            print("\n")




if __name__ == "__main__":
    # vector_store = VectorStore()
    # db_data = get_all_transactions()
    # vector_store.store_vectors2(db_data)
    # collections = vector_store.client.list_collections()
    # print(vector_store.client.get_collection("transactions").metadata)


    # print(vector_store.search_vectors("Sarjapur Road"))
    # print(vector_store.search_vectors("refId is 804813039157"))
    # print(vector_store.search_vectors("AXIS0000058"))

    # qdrant = QdrantVectorStore(collection_name='raw_data_search')
    qdrant = QdrantVectorStore()

    # qdrant.delete_collection()
    # qdrant.add_vectors()
    # qdrant.search_vector("Cennox Chain Leeds GB")
    # qdrant.search_vector("PI323")
    qdrant.search_vector('gaurav.kumar@gmailcom')






