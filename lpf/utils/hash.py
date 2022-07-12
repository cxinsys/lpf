import xxhash

    
def get_hash_digest(x):
    h = xxhash.xxh64()
    h.update(x)
    return h.intdigest()
