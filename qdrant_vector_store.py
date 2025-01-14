import copy
import json
import re
import traceback
import uuid
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, Iterable

import pandas as pd
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
from prettytable import PrettyTable


open_ai_embeds = OpenAIEmbeddings()


class QdrantVectorStore:

    def __init__(self,collection_name = "openaisearch",embedding_dimension=1536,query = ''):

        self.collection_name = collection_name

        self.query = query

        self.client = QdrantClient(url="http://localhost:6333", timeout=3000)


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


        self.data = get_all_transactions() if not self.query else  get_all_transactions(self.query)

    def formatted_json_str(self, json_str, default=False):
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

    def formatted_xml_str(self, xml_data):
        soup = BeautifulSoup(xml_data, "xml")

        data = soup.contents[0]

        def str_elements(data):
            st = '' + data.name
            for t in data.contents:
                if not isinstance(t, str):
                    att = t.attrs
                    st += self.formatted_json_str(json.dumps(att), default=True)
                    if t.contents:
                        st += " " + str_elements(t)

            return st + ' '

        return str_elements(data)

    def preprocess_xml(self, xml_str):
        # lines = xml_str.strip().splitlines()
        # if lines[0].startswith('<?xml'):
        #     lines = lines[1:]
        #
        # wrapped_xml = "<Root>" + "".join(lines) + "</Root>"
        return xml_str

    def convert_xml_to_json(self, xml_str):
        try:
            preprocessed_xml = self.preprocess_xml(xml_str)
            root = ET.fromstring(preprocessed_xml)
            result = {child.tag: child.attrib for child in root}
            return json.dumps(result, indent=4)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return self.formatted_xml_str(xml_str)

    def convert_payload(self, payload):
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

    def chunk_text_with_overlap(self, text, max_tokens=450, overlap=50):

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

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)


    def update_points(self, update_points=None):


        '''  update_points=[{"guid":"0029fe79-4994-4e85-a16b-d0d894850d0c","payload":{'date': datetime.datetime(2024, 8, 16, 5, 23, 2, 207000), 'endpoint': '/HB/IB/b369b83d-9a2d-4947-bc29-fb53930dfb00', 'guid': '0029fe79-4994-4e85-a16b-d0d894850d0c', 'request': '{\n  "productId": "1000",\n  "productName": "Iphone 15",\n  "company": "Apple",\n  "price": "85000"\n}', 'response': '{\n  "paymentId": "PI323",\n  "message": "Payment Successfull"\n}', 'source': 'hostbox'}}]
             '''

        if update_points is None:
            update_points = []
        for point in update_points:
            self.client.set_payload(collection_name=self.collection_name,payload=point['payload'], points=[point['guid']])



    def retrive_record(self, guids=None):

        if guids is None:
            guids = []

        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=guids)

        return results


    def add_vectors(self,preprocess = False):
        data = copy.deepcopy(self.data)
        vec = []
        guids = []
        payloads =[]
        for idx,transaction in enumerate(tqdm(data)):
            try:
                response = "Response " + transaction.get("response") if transaction.get("response") else ''
                request = "\n" + " Request " + transaction.get("request") if transaction.get("request") else ''
                document = response + request

                #ADD Chunking layer

                if preprocess:
                    document = self.convert_payload(document)


                if document:
                   vec.append(document.lower())
                   guids.append(transaction["guid"])
                   payloads.append({"text":document.lower(),"response": True, "text": document.lower(), "date": str(transaction.get("date")),"guid": transaction.get("guid"),"source":transaction.get("source")})


            except Exception as e:
                print(e)
                print(traceback.format_exc())
                return



        embeddings, texts = open_ai_embeds.generate_embeddings([i['text'] for i in payloads])


        # payloads = [payload for payload in payloads if payload.get("text") in texts]

        points = [
                PointStruct(id=doc['guid'], vector=v,
                            payload=doc)
                for id,(v, doc) in enumerate(tqdm(zip(embeddings, payloads)))
            ]

        operation_info = self.client.upload_points(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )



    def add_vectors_v2(self):

        data = copy.deepcopy(self.data)

        payloads = []

        df = pd.DataFrame(data)

        for _, row in df.iterrows():
            context = process_row(row)
            metadata=row.to_dict()
            metadata.update({"Text": context})
            payloads.append(metadata)

        embeddings, texts = open_ai_embeds.generate_embeddings([i['Text'] for i in payloads])

        points = [
            PointStruct(id=doc['id'], vector=v,
                        payload=doc)
            for id, (v, doc) in enumerate(tqdm(zip(embeddings, payloads)))
        ]

        operation_info = self.client.upload_points(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )






    def search_vector(self,query, top_k =10):
        query_embed,text =open_ai_embeds.generate_embeddings([query])
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embed[0],
            limit=top_k,
        ).points

        results = [{**i.payload, "score": i.score} for i in hits]

        sorted_results = sorted(results, key=lambda x: parse_date(x['date']),reverse=True)


        # df = pd.DataFrame(sorted_results)
        #
        #
        # df.to_excel("output.xlsx",index=False)

        final_results = []

        for hit in sorted_results:

            final_results.append({
            "guid": hit.get("guid"),
            "score": hit.get("score"),
            "date": hit.get("date"),
            "source": hit.get("source")})

        return final_results



    def search_vector_v2(self,query, top_k =10):
        query_embed,text =open_ai_embeds.generate_embeddings([query])
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embed[0],
            limit=top_k,
        ).points

        results = [{**i.payload, "score": i.score} for i in hits if i.score > 0.4]


        final_results = []

        df = pd.DataFrame(results)

        df.to_excel("output.xlsx",index=False)

        for hit in results:

            final_results.append({
                "id": hit.get("id"),
                "collection_name": hit.get("collection"),
                "request_name": hit.get("request_name"),
                "method":hit.get("method"),
                "body": hit.get("request"),
                "headerSize": json.loads(hit.get("settings", {})).get('headerSize', 2),
                "metadata": hit,
                "content_type": hit.get("content_type"),
                "packager_guid": hit.get("packager_guid"),
                "endpoint": hit.get("endpoint"),
                "responseContent":hit.get("response"),
                "headers":hit.get("headers",[]),
                "auth":hit.get("auth",{"authType": "none"}),
                "params":hit.get("params"),
                "score": hit.get("score")})

        return final_results




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
    # qdrant = QdrantVectorStore()
    # qdrant = QdrantVectorStore(collection_name='update_points')

    qdrant = QdrantVectorStore(collection_name='marketplace',query="SELECT * FROM marketplace")

    # qdrant.delete_collection()
    # qdrant.add_vectors_v2()
    # qdrant.update_points()

    # qdrant.search_vector("Cennox Chain Leeds GB and 1651412110")
    # qdrant.search_vector("PI323")
    # qdrant.search_vector('acct is 058010100083000')

    # print(qdrant.search_vector_v2("create request for deactivate of prutan"))

    # print(qdrant.search_vector_v2("create requests for auth request visa of 50 times"))

    # print(qdrant.search_vector_v2("create request for deactivate of razorpay"))

    # print(qdrant.search_vector_v2("create request to fetch the status of mastercard"))

    query = "create a registration request from prutan"
    print(qdrant.search_vector_v2(query=query))







