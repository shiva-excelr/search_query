import copy
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

from database import get_all_transactions

special_chars = [
    '#', '$', '%', '&', '*', '+', '-', ':',  '=', '@', '_',  '|',  '~','-'
]

def create_datset():
    def parse_iso8583(xml_data):
        """Parse ISO 8583 XML and extract key fields."""
        root = ET.fromstring(xml_data)
        fields = {field.attrib["id"]: field.attrib["value"] for field in root.findall("field")}
        return fields



    def create_finetuning_entry(row):
        """Convert SQL row into fine-tuning JSON format."""
        request_fields = parse_iso8583(row["request"])
        response_fields = parse_iso8583(row["response"])

        input_text = f"Process an ISO 8583 {row['request_name']} request with the following fields: {row}."
        output_text = f"ISO 8583 {row['request_name']} response: {response_fields}."

        return {"input": input_text, "output": output_text}


    # Example data (Replace with SQL extracted values)
    request_xml = """<isomsg> ... </isomsg>"""  # Place actual request XML
    response_xml = """<isomsg> ... </isomsg>"""  # Place actual response XML

    data = get_all_transactions(query = "SELECT * FROM marketplace")

    train_data = [create_finetuning_entry(row) for row in data]

    # Save as JSONL for LLaMA fine-tuning
    with open("finetuning_data.jsonl", "w") as f:
        json.dump(train_data, f)
        f.write("\n")





def create_datset_update():
    import json
    import sqlite3



    # Convert data to JSONL format for fine-tuning
    output_file = "update_fine_tune_data.jsonl"

    data = get_all_transactions(query = "SELECT * FROM marketplace")


    with open(output_file, "w", encoding="utf-8") as f:
        dat = []
        for row in data:

            id_, guid, collection, request_name, content_type, packager,request, response, rules, settins, description,teamguid,createdby, createdon, updatedby,updatedon = row

            conversation = [
                {"role": "system", "content": "You are a financial transaction processing assistant."},
                {"role": "user",
                 "content": f"Process the following {row['collection']} {row['request_name']} transaction request:\n{row['request']}"},
                {"role": "assistant", "content": f"The processed response is:\n{row['response']}"}
            ]

            dat.append(conversation)



        f.write(json.dumps({"messages": dat}, ensure_ascii=False) + "\n")

    print(f"Dataset saved as {output_file}")


def create_datset_update1():

    data = get_all_transactions(query = "SELECT * FROM marketplace")

    dat = []
    for row in data:
        id_, guid, collection, request_name, content_type, packager, request, response, rules, settins, description, teamguid, createdby, createdon, updatedby, updatedon = row

        d= dict(row)
        conversation = [
            {"role": "system", "content": "You are a financial transaction processing assistant."},
            {"role": "user",
             "content": f"Process the following {row['collection']} {row['request_name']} transaction request:\n{row['request']}"},
            {"role": "assistant", "content": f"The processed response is:\n{row['response']}"}
        ]

        dat.append(conversation)

    training_data = []

    for conversation in dat:
        # Construct the prompt by joining the system, user, and assistant messages
        prompt = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation])
        completion = conversation[-1]['content']  # Last entry is always the assistant's response

        # Append the structured example to training data
        training_data.append({
            "prompt": prompt,
            "completion": completion
        })

    # Save the data in JSONL format
    # with open("fine_tune_data.jsonl", "w") as f:
    #     for example in training_data:
    #         f.write(json.dumps(example) + "\n")

    with open("fine_tune_data.jsonl", "w") as f:
        json.dump(training_data, f)
        f.write("\n")




FORMAT = "alpaca"  # Change this as needed

# Convert SQL data into Alpaca Format
def convert_to_alpaca(data):

    dataset = []
    for d in data:

        desc = json.loads(d['description'])['desc']
        inpu =  f"Metadata: GUID= {d['guid']} collection={d['collection']}, \n request_name={d['request_name']}, \nContent Type={d['content_type']} \nPackager guid {d['packager_guid']} \nTeam Guid {d['team_guid']}"

        a= {
            "format": "alpaca",
            "instruction": f"Generate an ISO8583 for {d['request_name']} request for {d['collection']}",
            "input":'',
            "output": inpu + " "+f" Request for {d['collection']} {d['request_name']} is: \n'{d['request']}' \n and Response is '{d['response']}'"
        }
        dataset.append(a)

    return {"conversations":dataset}

# Convert SQL data into OpenAssistant Format
def convert_to_openassistant(data):
    return [
        {
            "messages": [
                {"role": "system", "content": "You are an expert in ISO8583 transaction processing."},
                {"role": "user", "content": "Generate an authorization request for a VISA transaction."},
                {"role": "assistant", "content": data["request"]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the response for the given request?"},
                {"role": "assistant", "content": data["response"]}
            ]
        }
    ]

# Convert SQL data into Self-Instruct Format
def convert_to_self_instruct(data):
    return [
        {
            "instruction": "Extract important fields from an ISO8583 authorization request.",
            "input": data["request"],
            "output": "Transaction Type: 0100, Amount: {{$generateAmount(4)}}, Card Number: {{$generateAlphaNumeric(19)}}"
        },
        {
            "instruction": "Translate the given ISO8583 message into human-readable format.",
            "input": data["request"],
            "output": "This is an authorization request for a VISA card transaction."
        }
    ]


# print(create_datset())


sql_data = get_all_transactions(query = "SELECT * FROM marketplace")
# if FORMAT == "alpaca":
#     dataset = convert_to_alpaca(sql_data)
# elif FORMAT == "openassistant":
#     dataset = convert_to_openassistant(sql_data)
# elif FORMAT == "self_instruct":
#     dataset = convert_to_self_instruct(sql_data)
# else:
#     raise ValueError("Invalid format selection!")

def generate_questions(collection_name, request_name, content_type):
    request_name = request_name.title()
    return [
        f"What is the {request_name} request format for {collection_name}?",
        f"Generate an {content_type} request for {request_name} in {collection_name}.",
        f"Can you provide request details for the {request_name} transaction for {collection_name}",
        f"What is the expected request for a {request_name} for {collection_name}",
        f"Explain the key fields in an request for {request_name} of {collection_name}",
        f"Retrieve the request message structure for {request_name} in {collection_name}.",
        f"What is the purpose of the payload in {request_name} of {collection_name} requests?",
        f"Describe the payload for {request_name} of {collection_name}"
        f"Can you provide GUID for the {request_name} transaction of {collection_name}?",
       f"Could you share the GUID for the {request_name} transaction under the {collection_name} collection?",
        f"What is the GUID for the {request_name} transaction in the {collection_name} collection?",
        f"Can you give me the GUID associated with the {request_name} transaction from the {collection_name} collection?",
        f"Please provide the GUID for the {request_name} transaction within the {collection_name} collection.",
        f"I'm looking for the GUID for the {request_name} transaction of {collection_name}. Can you provide that?",
        f"What GUID corresponds to the {request_name} transaction in the {collection_name} collection?",
        f"Can you tell me the GUID for the {request_name} transaction from {collection_name}?",
        f"Could you provide the GUID for the {request_name} request in the {collection_name}?",
        f"Would you be able to supply the GUID for the {request_name} transaction within {collection_name}?",
        f"Do you have the GUID for the {request_name} transaction of the {collection_name} collection?",

    ]



def generate_questions2(collection_name, request_name, content_type):
    request_name = request_name.title()

    query_templates = [
        f"Show me a request for {request_name} for {collection_name}",
        f"Give me an {request_name} request for {collection_name}",
        f"Find me a request related to {request_name} for {collection_name}",
        f"Fetch the request details for {request_name} under {collection_name}",
       f"Retrieve a {request_name} request from {collection_name}",
        f"I need the exact request for {request_name} in {collection_name}",
        f"List a request for {request_name} associated with {collection_name}",
        f"What is the request payload for {request_name} in {collection_name}?",
        f"Provide me with the full details of {request_name} request in {collection_name}",
        f"Do you have a record of a {request_name} request in {collection_name}?",
        f"Find the request payload for {request_name} in {collection_name}",
        f"Get me the API request structure for {request_name} under {collection_name}",
        f"Show the complete request for {request_name} inside {collection_name}",
        f"I want the  request for {request_name} in {collection_name}",
        f"Extract the request payload for {request_name} from {collection_name}",
        f"Display the request details of {request_name} from {collection_name}",
        f"Locate the API request for {request_name} under {collection_name}",
        f"Fetch request data for {request_name} in {collection_name}",
        f"Find a {request_name} API call within {collection_name}",
        f"Can you give me the exact {request_name} request from {collection_name}?",
        f"What GUID corresponds to the {request_name} transaction in the {collection_name} collection?",
        f"Find the GUID for the {request_name} request in the {collection_name} collection.",
        f"Retrieve the GUID linked to the {request_name} API request under {collection_name}.",
        f"Which GUID is associated with the {request_name} request in {collection_name}?",
        f"Give me the GUID for the {request_name} transaction stored in {collection_name}.",
        f"Find me the exact GUID for the {request_name} request inside {collection_name}.",
        f"Fetch the GUID that corresponds to the {request_name} operation under {collection_name}.",
        f"Locate the GUID for the {request_name} request within {collection_name}.",
        f"Provide the GUID assigned to the {request_name} transaction in {collection_name}.",
        f"Get me the GUID corresponding to {request_name} inside {collection_name}."
    ]

    return query_templates




def parse_iso8583(xml_data):
    root = ET.fromstring(xml_data)
    fields = {field.attrib["id"]: field.attrib["value"] for field in root.findall("field")}
    return fields

def create_llm_data(rows):
    conversations = defaultdict(list)
    dataset = {"conversations": []}

    SYSTEM_PROMPT = "You are an expert in retrieving ISO request IDs based on request names and collections."


    for row in rows:

        row = dict(row).values()

        id_, guid, collection_name, request_name, content_type, package_id, request_payload, response, rules, settings, description, team_guid, createdby, createdon, updatedby, updatedon = row

        request_payload = parse_iso8583(request_payload)
        response = parse_iso8583(response)

        description = json.loads(description)['desc']

        questions = generate_questions(collection_name, request_name, content_type)
        conversation_messages = []

        for question in questions:
            conversation_messages.append({"role": "user", "content": question})
            if "metadata" in question.lower():
                assistant_response = {
                    "role": "assistant",
                    "content": f"Metadata:Description = {description},\nCollection = {collection_name},\nRequest Name = {request_name},\nContent Type = {content_type},\nPackager GUID = {package_id},\nTeam GUID = {team_guid}\n"
                }

            elif "response" in question.lower():
                assistant_response = {
                    "role": "assistant",
                    "content": f"Expected response for {request_name}:\n{response}"
                }
            elif "request" in question.lower():
                assistant_response = {
                    "role": "assistant",
                    "content": f"Request details for {request_name}:\n{request_payload}"
                }
            elif "ISO8583" in question:
                assistant_response = {
                    "role": "assistant",
                    "content": f"Here is an example of an {content_type} request for {request_name} in {collection_name}:\n{request_payload}"
                }
            elif "Packager GUID" in question:
                assistant_response = {
                    "role": "assistant",
                    "content": "The Packager GUID is a unique identifier assigned to the ISO8583 packager module responsible for encoding and decoding transaction messages."
                }
            elif "guid" in question.lower():
                assistant_response = {
                    "role": "assistant",
                    "content": f"GUID for {request_name}:\n{guid}"
                }
            else:
                assistant_response = {
                    "role": "assistant",
                    "content": "This request follows the standard structure and validation for secure processing in ISO8583 transactions."
                }

            conversation_messages.append(assistant_response)

        dataset["conversations"].append({"system": SYSTEM_PROMPT, "messages": conversation_messages})

        # # Format assistant response with metadata
        # assistant_response = (
        #     f"Metadata: GUID= {guid}, collection={collection_name}, request_name={request_name}, Content Type={content_type}, "
        #     f"Packager guid={package_id}, Team Guid={team_guid}.\n\nRequest:\n{request_payload}\n\nResponse:\n{response}"
        # )
        #
        # conversations[guid].append({
        #     "role": "user",
        #     "content": f"Generate a {content_type} request for {collection_name}, request type: {request_name}."
        # })
        #
        # conversations[guid].append({
        #     "role": "assistant",
        #     "content": assistant_response
        # })

    # formatted_data = []
    #
    # for guid, messages in conversations.items():
    #     formatted_data.append({
    #         "system": "You are a financial transactions retrieval assistant specializing in payment processing techniques of ISO of formats 1987 SPEC , 1993 SPEC, 2003 SPEC and helping users generate guids, request, response payloads and also help in solving and resolving any problems in formatting the ISO formats of different specs, providing the description of each spec value and details if asked",
    #         "messages": messages
    #     })



    # Save as JSON file
    with open("llm_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print("✅ JSON dataset saved as dataset.json")

def create_llm_data1(rows):
    conversations = defaultdict(list)
    dataset = {"conversations": []}


    SYSTEM_PROMPT = "You are an expert in retrieving ISO request names and collections from the user and return Metadata for the intent of the request"


    for row in rows:

        row = dict(row).values()

        id_, guid, collection_name, request_name, content_type, package_id, request_payload, response, rules, settings, description, team_guid, createdby, createdon, updatedby, updatedon = row

        # request_payload = parse_iso8583(request_payload)
        # response = parse_iso8583(response)

        description = json.loads(description)['desc']

        questions = generate_questions(collection_name, request_name, content_type)
        conversation_messages = []


        for question in questions:
            random_char = random.choice(special_chars)
            cpy_guid = copy.deepcopy(guid)
            cpy_guid = cpy_guid.replace("-",random_char*2)
            conversation_messages.append({"from": "human", "value": question})
            assistant_response = {
            "from": "gpt",
            "value": f"Description = {description},\nCollection = {collection_name},\nRequest Name = {request_name},\nContent Type = {content_type}, Request = <PrutanID value = \"{cpy_guid}\">{request_payload}"
        }


            conversation_messages.append(assistant_response)

        # dataset["conversations"].append({"system": SYSTEM_PROMPT, "messages": conversation_messages})
        dataset["conversations"].append(conversation_messages)




    # Save as JSON file
    with open("llm_dataset_all.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print("✅ JSON dataset saved as dataset.json")


def create_llm_data2(rows):
    conversations = defaultdict(list)
    dataset = {}

    q=[]
    a=[]


    SYSTEM_PROMPT = "You are an expert in retrieving ISO request names and collections from the user and return Metadata for the intent of the request"


    for row in rows:

        row = dict(row).values()

        id_, guid, collection_name, request_name, content_type, package_id, request_payload, response, rules, settings, description, team_guid, createdby, createdon, updatedby, updatedon = row

        request_payload = parse_iso8583(request_payload)
        response = parse_iso8583(response)

        description = json.loads(description)['desc']

        questions = generate_questions(collection_name, request_name, content_type)
        conversation_messages = []

        for question in questions:
            # conversation_messages.append({"from": "human", "value": question})
            q.append(question)
            a.append( f"Metadata:GUID= {guid}, Description = {description},\nCollection = {collection_name},\nRequest Name = {request_name},\nContent Type = {content_type},\nPackager GUID = {package_id},\nTeam GUID = {team_guid}\n, Request = {request_payload}, Response = {response}")
        #     assistant_response = {
        #     "from": "gpt",
        #     "value": f"Metadata:GUID= {guid}, Description = {description},\nCollection = {collection_name},\nRequest Name = {request_name},\nContent Type = {content_type},\nPackager GUID = {package_id},\nTeam GUID = {team_guid}\n, Request = {request_payload}, Response = {response}"
        # }


            # conversation_messages.append(assistant_response)

        # dataset["conversations"].append({"system": SYSTEM_PROMPT, "messages": conversation_messages})
        # dataset["conversations"].append(conversation_messages)


    dataset['questions'] =q
    dataset['answers'] = a

    # Save as JSON file
    with open("llm_dataset_all2.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print("✅ JSON dataset saved as dataset.json")


def create_llm_data3(rows):


    data= []

    dataset = {"conversations": data}



    for row in rows:

        row = dict(row).values()

        id_, guid, collection_name, request_name, content_type, package_id, request_payload, response, rules, settings, description, team_guid, createdby, createdon, updatedby, updatedon = row



        description = json.loads(description)['desc']

        questions = generate_questions2(collection_name, request_name, content_type)

        for id,question in enumerate(questions):
            cpy_guid = copy.deepcopy(guid)


            messages = [{"role": "system","content": "You are a strict assistant that only provides exact matches from the dataset. You do not assume or generate unknown data."}, {"role": "user", "content": question},{"role": "assistant",
                     "content":f"<PrutanID>\"{cpy_guid}\"<\PrutanID>, <Description> {description} <\Description>,<Collection> {collection_name} <\Collection>,<Request Name> {request_name} <\Request Name>,<Content Type> {content_type} <\Content Type>, <Request> {request_payload} <\Request>" }]

            # else:
            #     messages.append({"role": "user", "content": question}, {"role": "assistant",
            #                                                             "content": f"<PrutanID>\"{cpy_guid}\"<\PrutanID>, <Description> {description} <\Description>,<Collection> {collection_name} <\Collection>,<Request Name> {request_name} <\Request Name>,<Content Type> {content_type} <\Content Type>, <Request> {request_payload} <\Request>"})







            data.append(messages)

            if len(data) == 500:
                pass






    # Save as JSON file
    with open("llm_dataset_30-1-25.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print("✅ JSON dataset saved as dataset.json")

print(create_llm_data3(sql_data))
#
# output_file = f"fine_tune_{FORMAT}.json"
# with open(output_file, "w") as f:
#     json.dump(dataset, f, indent=4)
#
# print(f"Dataset saved to {output_file}")


