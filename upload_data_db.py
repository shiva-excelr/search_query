import mysql.connector
import uuid
from mysql.connector import connect
import random
from itertools import product

api_keywords = [
    "auth", "login", "register", "logout", "create", "update", "delete", "fetch",
    "retrieve", "query", "search", "get", "post", "put", "patch", "validate",
    "upload", "download", "activate", "deactivate", "verify", "reset",
    "purchase", "cancel", "status"
]

collec = ["VISA", "MASTERCARD", "RAZORPAY", "UPI", "PRUTAN"]

# Precompute all unique combinations of new_collection and new_request_name
unique_combinations = list(product(collec, api_keywords))
random.shuffle(unique_combinations)  # Shuffle for randomness

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
        else:
            # Use the value from the original row
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
