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