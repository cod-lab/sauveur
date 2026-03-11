
class Default_Env_Vars:   # default values are below, user can pass their values
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 0
    MODEL_PROVIDER = 'huggingface'
    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'#all-MiniLM-L6-v2'
    MODEL_DIMENSION = 768       # its not required with huggingface, bcz all its models hv already fixed dimension
    MODEL_PROVIDER_KWARGS = {
        'encode_kwargs': {"normalize_embeddings": True}
    }
    NO_OF_DOCS_PER_BULK_OBJECT = 5#1000     # bydefault max no. of docs allowed to upload at once thru opensearch bulk api
    K = 10
    TOP = 10



