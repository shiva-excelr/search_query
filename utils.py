import hashlib

def generate_hash_key(text:str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()