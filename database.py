import mysql.connector
from mysql.connector import connect

def get_sql_connection():
    return connect(
        host="localhost",
        user="root",
        password="password",
        database="common_analytics"
    )

def get_all_transactions():
    connection = get_sql_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT createdOn as date,guid, response, request FROM common_analytics ORDER BY createdOn DESC LIMIT 1000")
    # cursor.execute(
    #     "SELECT guid, response, request FROM common_analytics WHERE LOWER(guid) = LOWER('0029fe79-4994-4e85-a16b-d0d894850d0c')")
    records = cursor.fetchall()
    cursor.close()
    connection.close()
    return records


