import json

import mysql.connector
import uuid
from mysql.connector import connect
import random
from itertools import product
api_keywords = [
    "auth", "login", "register", "logout", "create payment", "update payment", "fetch order",
    "terminate", "search order", "validate payment","activate user", "deactivate user", "verify payment", "purchase", "cancel", "status of payment"
]
api_keywords2 = [
     "login", "register", "logout", "create payment", "update payment", "fetch order",
    "terminate", "search order", "validate payment","activate user", "deactivate user", "verify payment", "cancel", "status of payment"
]


collec = [ "RAZORPAY", "UPI","PAYPAL"]

collec2 = ["VISA", "MASTERCARD"]


coll= {"UPI":["collect payment","generate Qr","capture UPIId", "check VPA","status check","payment confirmation"],
       "MASTERCARD":["auth","purchase","register", "delete","retreive scopes","verify identity", "fetch user identity"],
       "VISA":["auth","purchase","register", "delete","retreive scopes","verify identity", "fetch user identity"],
       "RAZORPAY":["create customer","auth","fetch orders","fetch payment downtime details","update payment"],
       "PAYPAL":["terminate","auth","user info","generate access","create order", "show order", "authorize payment"]
       }


responses  = {"UPI":'''<isomsg>
  <!-- org.jpos.iso.packager.GenericPackager -->
  <header>323434301020000044432308489443000000a02434322</header>
  <field id="0" value="0100"/>
  <field id="2" value="user@upi"/>
  <field id="3" value="000000"/>
  <field id="4" value="200.00"/>
  <field id="6" value="200.00"/>
  <field id="7" value="29-01-2025 10:45:00"/>
  <field id="10" value="61000000"/>
  <field id="11" value="144896"/>
  <field id="12" value="113809"/>
  <field id="13" value="29-01"/>
  <field id="15" value="29-01"/>
  <field id="18" value="5999"/>
  <field id="19" value="IN"/>
  <field id="22" value="071"/>
  <field id="25" value="51"/>
  <field id="32" value="00333333"/>
  <field id="33" value="00333333"/>
  <field id="35" value="user@upi=1225251000000012345"/>
  <field id="37" value="897465901234"/>
  <field id="41" value="NWG00003"/>
  <field id="42" value="UPIGateway"/>
  <isomsg id="43">
    <field id="1" value="SYNAP"/>
    <field id="2" value="ONTRAIO"/>
    <field id="3" value="CA"/>
  </isomsg>
  <field id="48" value="5CE385A2A3"/>
  <field id="49" value="INR"/>
  <field id="51" value="INR"/>
  <isomsg id="62">
    <field id="2" value="153621623779500"/>
  </isomsg>
  <isomsg id="63">
    <field id="1" value="0000"/>
    <field id="19" value="500"/>
  </isomsg>
</isomsg>
''',
       "MASTERCARD":'''<isomsg>
  <!-- org.jpos.iso.packager.GenericPackager -->
  <header>2343242320000232323432430000000000088484049944</header>
  <field id="0" value="0100"/>
  <field id="2" value="5500000000000004"/>
  <field id="3" value="000000"/>
  <field id="4" value="50.00"/>
  <field id="6" value="50.00"/>
  <field id="7" value="29-01-2025 10:45:00"/>
  <field id="10" value="61000000"/>
  <field id="11" value="144896"/>
  <field id="12" value="113809"/>
  <field id="13" value="29-01"/>
  <field id="15" value="29-01"/>
  <field id="18" value="5999"/>
  <field id="19" value="IN"/>
  <field id="22" value="071"/>
  <field id="25" value="51"/>
  <field id="32" value="00333333"/>
  <field id="33" value="00333333"/>
  <field id="35" value="5500000000000004=1122233000000012345"/>
  <field id="37" value="897465901234"/>
  <field id="41" value="NWG00002"/>
  <field id="42" value="MastercardGateway"/>
  <isomsg id="43">
    <field id="1" value="SYNAP"/>
    <field id="2" value="ONTRAIO"/>
    <field id="3" value="CA"/>
  </isomsg>
  <field id="48" value="5CE385A2A3"/>
  <field id="49" value="USD"/>
  <field id="51" value="USD"/>
  <isomsg id="62">
    <field id="2" value="153621623779500"/>
  </isomsg>
  <isomsg id="63">
    <field id="1" value="0000"/>
    <field id="19" value="500"/>
  </isomsg>
</isomsg>
''',
       "RAZORPAY":'''<isomsg>
  <!-- org.jpos.iso.packager.GenericPackager -->
  <header>1231213200000100230100010100000121342100000</header>
  <field id="0" value="0100"/>
  <field id="2" value="razorpay_order_id"/>
  <field id="3" value="000000"/>
  <field id="4" value="500.00"/>
  <field id="6" value="500.00"/>
  <field id="7" value="29-01-2025 10:45:00"/>
  <field id="10" value="61000000"/>
  <field id="11" value="144896"/>
  <field id="12" value="113809"/>
  <field id="13" value="29-01"/>
  <field id="15" value="29-01"/>
  <field id="18" value="5999"/>
  <field id="19" value="IN"/>
  <field id="22" value="071"/>
  <field id="25" value="51"/>
  <field id="32" value="00333333"/>
  <field id="33" value="00333333"/>
  <field id="35" value="razorpay_order_id=1225251000000012345"/>
  <field id="37" value="897465901234"/>
  <field id="41" value="NWG00004"/>
  <field id="42" value="RazorpayGateway"/>
  <isomsg id="43">
    <field id="1" value="SYNAP"/>
    <field id="2" value="ONTRAIO"/>
    <field id="3" value="CA"/>
  </isomsg>
  <field id="48" value="5CE385A2A3"/>
  <field id="49" value="INR"/>
  <field id="51" value="INR"/>
  <isomsg id="62">
    <field id="2" value="153621623779500"/>
  </isomsg>
  <isomsg id="63">
    <field id="1" value="0000"/>
    <field id="19" value="500"/>
  </isomsg>
</isomsg>
''',
       "PAYPAL":'''<isomsg>
  <!-- org.jpos.iso.packager.GenericPackager -->
  <header>12122432423008200383340000007868980334324234343987970</header>
  <field id="0" value="0100"/>
''',
     "VISA":'''<isomsg>
  <!-- org.jpos.iso.packager.GenericPackager -->
  <header>22010200000000000000000000000000000000000000</header>
  <field id="0" value="0100"/>
  <field id="2" value="{{$generateAlphaNumeric(19)}}"/>
  <field id="3" value="{{$randomNumber(6)}}"/>
  <field id="4" value="{{$generateAmount(4)}}"/>
  <field id="6" value="{{$generateAmount(4)}}"/>
  <field id="7" value="{{$generateDate(dd-MM-yyyy hh:mm:ss,+1days,-1month)}}"/>
  <field id="10" value="61000000"/>
  <field id="11" value="144896"/>
  <field id="12" value="113809"/>
  <field id="13" value="{{$generateDate(ddMM)}}"/>
  <field id="15" value="{{$generateDate(ddMM)}}"/>
  <!-- <field id="16" value="0308"/> -->
  <!-- <field id="18" value="5734"/> -->
  <field id="18" value="5999"/>
  <field id="19" value="{{$generateCountryCode(India)}}"/>
  <field id="22" value="071"/>
  <field id="25" value="51"/>
  <field id="32" value="00333333"/>
  <field id="33" value="00333333"/>
  <field id="35" value="04192930040053694=2803201000000497268"/>
  <field id="37" value="{{$generateRandomNumber(12)}}"/>
  <field id="41" value="NWG00001"/>
  <field id="42" value="CardAcceptorCod"/>
  <isomsg id="43">
    <field id="1" value="SYNAP"/>
    <field id="2" value="ONTRAIO"/>
    <field id="3" value="CA"/>
  </isomsg>
  <field id="48" value="5CE385A2A3"/>
  <field id="49" value="{{$generateCurrencyCode(India)}}"/>
  <field id="51" value="{{$generateCurrencyCode(India)}}"/>
  <!-- <field id="61" value="000000000000000000000000000004006001"/> -->
  <isomsg id="62">
    <field id="2" value="153621623779500"/>
  </isomsg>
  <isomsg id="63">
    <!-- org.jpos.iso.packager.Base1SubFieldPackager -->
    <field id="1" value="0000"/>
    <field id="19" value="500"/>
  </isomsg>
</isomsg>
'''
       }

# Precompute all unique combinations of new_collection and new_request_name
unique_combinations1 = list(product(collec, api_keywords))
unique_combinations2 = list(product(collec2, api_keywords2))
unique_combinations = unique_combinations1+ unique_combinations2
random.shuffle(unique_combinations)  # Shuffle for randomness

import random
import string
from datetime import datetime, timedelta


def random_string(length=16):
    """Generate a random alphanumeric string of a given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_number(length=12):
    """Generate a random number of a given length."""
    return ''.join(random.choices(string.digits, k=length))


def random_amount():
    """Generate a random amount with two decimal places."""
    return round(random.uniform(1, 1000), 2)


def random_date(start_date, end_date):
    """Generate a random date between two given dates."""
    delta = end_date - start_date
    random_day = random.randint(0, delta.days)
    return start_date + timedelta(days=random_day)


def random_currency_code():
    """Generate a random 3-character currency code."""
    return random.choice(["INR", "USD", "EUR", "GBP", "AUD"])


def random_upi_id():
    """Generate a random UPI ID."""
    return f"user{random.randint(1000, 9999)}@upi"


def random_header():
    """Generate a random header for ISO message."""
    return ''.join(random.choices(string.hexdigits.lower(), k=40))


def generate_iso_message(payment_type):
    """Generate a random ISO message for the given payment type."""

    # Date Range for random date generation
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    header = random_header()

    # Customize fields based on payment type
    if payment_type.lower() == "visa":
        merchant_code = "VisaPaymentGateway"
        currency_code = "INR"
        additional_field = "<field id=\"47\" value=\"VisaProcessingInfo\"/>"
    elif payment_type.lower() == "mastercard":
        merchant_code = "MastercardPaymentGateway"
        currency_code = "USD"
        additional_field = "<field id=\"47\" value=\"MastercardProcessingInfo\"/>"
    elif payment_type.lower() == "upi":
        merchant_code = "UPIPaymentGateway"
        currency_code = "INR"
        additional_field = f"<field id=\"48\" value=\"{random_upi_id()}\"/>"
    elif payment_type.lower() == "razorpay":
        merchant_code = "RazorpayPaymentGateway"
        currency_code = "INR"
        additional_field = "<field id=\"47\" value=\"RazorpayProcessingInfo\"/>"
    elif payment_type.lower() == "paypal":
        merchant_code = "PayPalPaymentGateway"
        currency_code = "USD"
        additional_field = "<field id=\"47\" value=\"PayPalProcessingInfo\"/>"
    else:
        merchant_code = "UnknownPaymentGateway"
        currency_code = "USD"
        additional_field = ""

    iso_message = f"""<isomsg>
  <header>{header}</header>
  <field id="0" value="0100"/>
  <field id="2" value="{random_string(16)}"/>
  <field id="3" value="{random_number(6)}"/>
  <field id="4" value="{random_amount()}"/>
  <field id="6" value="{random_amount()}"/>
  <field id="7" value="{random_date(start_date, end_date).strftime('%d-%m-%Y %H:%M:%S')}"/>
  <field id="10" value="61000000"/>
  <field id="11" value="{random_number(6)}"/>
  <field id="12" value="{random_number(6)}"/>
  <field id="13" value="{random_date(start_date, end_date).strftime('%d-%m')}"/>
  <field id="15" value="{random_date(start_date, end_date).strftime('%d-%m')}"/>
  <field id="18" value="5999"/>
  <field id="19" value="IN"/>
  <field id="22" value="071"/>
  <field id="25" value="51"/>
  <field id="32" value="{random_number(8)}"/>
  <field id="33" value="{random_number(8)}"/>
  <field id="35" value="{random_string(16)}={random_string(16)}"/>
  <field id="37" value="{random_number(12)}"/>
  <field id="41" value="NWG00001"/>
  <field id="42" value="{merchant_code}"/>
  <field id="48" value="{random_string(16)}"/>
  <field id="49" value="{currency_code}"/>
  <field id="51" value="{currency_code}"/>
  {additional_field}
  <isomsg id="62">
    <field id="2" value="{random_number(15)}"/>
  </isomsg>
  <isomsg id="63">
    <field id="1" value="0000"/>
    <field id="19" value="500"/>
  </isomsg>
</isomsg>"""

    return iso_message


# Database connection
conn = connect(
    host="localhost",
    user="root",
    password="password",
    database="common_analytics"
)
cursor = conn.cursor()

# Define the ID of the row to duplicate
row_id_to_duplicate = 1

# Fetch the original row
cursor.execute("SELECT * FROM marketplace WHERE id = %s", (row_id_to_duplicate,))
original_row = cursor.fetchone()

if not original_row:
    print(f"No row found with id {row_id_to_duplicate}")
    conn.close()
    exit()

# Fetch column names dynamically
cursor.execute("SHOW COLUMNS FROM marketplace;")
columns_info = cursor.fetchall()
columns = [col[0] for col in columns_info]

# Prepare columns for insertion
columns_to_insert = [col for col in columns]

# Create a placeholder for the prepared statement
placeholders = ", ".join(["%s"] * len(columns_to_insert))

# Start ID from 101
starting_id = 101

# Create new rows
new_rows = []
for i in range(100):
    new_id = starting_id + i
    new_guid = str(uuid.uuid4())  # Generate a new GUID for each row

    # Use the precomputed unique combination
    if i < len(unique_combinations):
        new_collection, new_request_name = unique_combinations[i]
    else:
        print("Not enough unique combinations to generate 100 rows.")
        break

    # Prepare the new row data
    new_row = []
    for col in columns_to_insert:
        if col == "id":
            new_row.append(new_id)
        elif col == "guid":
            new_row.append(new_guid)
        elif col == "collection":
            new_row.append(new_collection)
        elif col == "request_name":
            new_row.append(new_request_name)
        elif col == "team_guid":
            val =  str(uuid.uuid4())
            new_row.append(val)
        elif col == "packager_guid":
            val =  str(uuid.uuid4())
            new_row.append(val)
        elif col == "request":
            payment_type = new_row[2]
            val = generate_iso_message(payment_type.lower())
            new_row.append(val)
        else:
            # Use the value from the original row
            if col == "description":
                original_value = json.dumps({"desc": f"{new_row[3]} request for {new_row[2]}"})
                new_row.append(original_value)
            else:
               original_value = original_row[columns.index(col)]
               new_row.append(original_value)

    new_rows.append(tuple(new_row))

# Insert new rows into the table
try:
    cursor.executemany(
        f"INSERT INTO marketplace ({', '.join(columns_to_insert)}) VALUES ({placeholders})",
        new_rows
    )
    conn.commit()
    print(f"Inserted {len(new_rows)} new rows starting with id {starting_id}.")
except mysql.connector.Error as e:
    print(f"An error occurred: {e}")
finally:
    cursor.close()
    conn.close()
