import hashlib
from datetime import datetime

def generate_hash_key(text:str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def parse_date(date_str):
    """Parse date with multiple possible formats."""
    formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format not supported: {date_str}")




def process_row(row):

    context = (
        f"The API request belongs to the collection '{row['collection']}' and is named '{row['request_name']}'. "
        f"It processes a '{row['request_name']}' request with the content type '{row['content_type']}'. "
        f"The request is identified by packager ID '{row['packager_guid']}' and is associated with team ID '{row['team_guid']}'. "
        f"Settings for this request are specified as '{row['settings']}'. "
        f"The request body is: {row['request']}. "
        f"The response for the request is: {row['response']}. "
        f"The description of the request is: {row['description']}."
    )
    return context