import xmltodict
from bs4 import BeautifulSoup
import json

def formatted_xml_str(xml_data):
    soup = BeautifulSoup(xml_data, "xml")

    data = soup.contents[0]

    def str_elements(data):
        st = '' + data.name
        for t in data.contents:
            if not isinstance(t, str):
                att = t.attrs
                st += formatted_json_str(json.dumps(att), default=True)
                if t.contents:
                    st += " " + str_elements(t)

        return st + ' '

    return str_elements(data)

def formatted_json_str(json_str,default=False):
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

data = """<?xml version="1.0" encoding="UTF-8"?>
<ns2:RespAuthDetails
	xmlns:ns2="http://npci.org/upi/schema/">
	<Head msgId="BOIa4097f0d7c684ca4a6e2eddc965968b1" orgId="410005"
ts="2018-02-17T13:39:56.040+05:30" ver="2.0"/>
	<Resp reqMsgId="1GRDpegBbA5wfscXLm20" result="SUCCESS"/>
	<Txn custRef="804813039157" id="AXIb1fbc9cea1f34049904e083034723d49"
initiationMode="00" note="testpay" refId="804813039157"
refUrl="http://axis.com/upi" ts="2018-02-17T13:39:54.944+05:30" type="PAY">
		<RiskScores/>
	</Txn>
	<Payer addr="ram@axis" code="0000" name="ram" seqNum="1" type="PERSON">
		<Info>
			<Identity id="058010100083492" type="ACCOUNT" verifiedName="Ram"/>
			<Rating verifiedAddress="TRUE"/>
		</Info>
		<Ac addrType="ACCOUNT">
			<Detail name="ACTYPE" value="SAVINGS"/>
			<Detail name="ACNUM" value="058010100083000"/>
			<Detail name="IFSC" value="AXIS0000058"/>
		</Ac>
		<Amount curr="INR" value="2.00"/>
	</Payer>
	<Payees>
		<Payee addr="laxmi@boi" code="0000" name="Laxmi"
seqNum="1" type="PERSON">
			<Info>
				<Identity id="910010050136217" type="ACCOUNT" verifiedName="Laxmi "/>
				<Rating verifiedAddress="TRUE"/>
			</Info>
			<Ac addrType="ACCOUNT">
				<Detail name="ACTYPE" value="SAVINGS"/>
				<Detail name="ACNUM" value="910010050136000"/>
				<Detail name="IFSC" value="BKID0000004"/>
			</Ac>
			<Amount curr="INR" value="2.00"/>
		</Payee>
	</Payees>
</ns2:RespAuthDetails>"""


data_dict = xmltodict.parse(data)
print(data_dict)
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

flat_data = flatten_dict(data_dict)
print(flat_data)


