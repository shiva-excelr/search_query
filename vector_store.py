import json
import re
import traceback
import uuid
import xml.etree.ElementTree as ET
from enum import Enum

from bs4 import BeautifulSoup
from datetime import datetime
import chromadb
from embeddings import generate_embedding
from base_wrapper import  *
from embeddings import ollama_ef

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
                return formatted_xml
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

if __name__ == "__main__":
    vector_store = VectorStore()
    #
    from database import get_all_transactions
    # db_data = get_all_transactions()
    # vector_store.store_vectors2(db_data)
    collections = vector_store.client.list_collections()
    print(vector_store.client.get_collection("transactions").metadata)


    # print(vector_store.search_vectors("Sarjapur Road"))
    # print(vector_store.search_vectors("refId is 804813039157"))
    print(vector_store.search_vectors("AXIS0000058"))

    # print(vector_store.client.get)



