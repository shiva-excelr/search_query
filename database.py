import mysql.connector
from mysql.connector import connect

def get_sql_connection():
    return connect(
        host="localhost",
        user="root",
        password="password",
        database="common_analytics"
    )

def get_all_transactions(query="SELECT createdOn as date,guid, response, request,source FROM common_analytics ORDER BY createdOn DESC"):
    connection = get_sql_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    # cursor.execute(
    #     "SELECT guid, response, request FROM common_analytics WHERE LOWER(guid) = LOWER('0029fe79-4994-4e85-a16b-d0d894850d0c')")
    records = cursor.fetchall()
    cursor.close()
    connection.close()
    return records


if __name__ == "__main__":
    data = get_all_transactions()
    res = []
    pay =[]
    for dat in data:
        if dat['response'] and dat['response'].startswith(("<xml","<?xml")):
            res.append(dat['response'])
        if dat['request'] and dat['request'].startswith(("<xml","<?xml")):
            pay.append(dat['request'])

    combined_list = res+pay
    import json

    with open('outputfile.json', 'w') as fout:
        json.dump(combined_list, fout)

