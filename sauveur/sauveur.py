from typing import Any, Optional
import json
from pprint import pprint as pp

from pydantic import PositiveInt, PositiveFloat
from opensearchpy import OpenSearch
from langchain_core.prompts import ChatPromptTemplate


from .helpers.errors_messages import Error_Messages
from .helpers.reponse_status_messages import Response_Status_Messages


class RAG:
    def __init__(self, api_key:str='', plan:str='free') -> None:
        """
        """
        self._vector_db = None
        self._rds = None


    def create_chunks(self,
        items: list[
            str
            |list[str|list|tuple|dict]
            |tuple[str|list|tuple|dict]
            |dict[str,str|list|tuple|dict]
        ],
        chunk_size: PositiveInt = 1500,
        chunk_overlap: PositiveInt = 0
    ) -> list[dict[str,Any]]:
        """
        Creates chunks from the given list of items. Each item can be a string, list, tuple or dict. The function uses langchain's RecursiveCharacterTextSplitter for string type items and RecursiveJsonSplitter for list and dict type items to create chunks.

        Args:
            items (list[str|list|tuple|dict]): List of items to be chunked.
            chunk_size (int): Size of each chunk. Default is 1500.
            chunk_overlap (int): Overlap between chunks. Default is 0.
        Returns:
            dict: List of items with their chunks and chunking status
        """
        all_chunks = []

        from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

        string_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        json_splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)

        for i,item in enumerate(items,start=1):
            ele = {
                'item_no': i,
                'item_type': type(item).__name__,
                'item': item,
                'chunking_status': None,
                'chunks': None,
                'chunking_error': None,
            }

            if isinstance(item, str):
                ele['chunks'] = string_splitter.split_text(item)
                ele['chunking_status'] = Response_Status_Messages.SUCCESS.value
            elif isinstance(item, (list, dict)):
                ele['chunks'] = json_splitter.split_json(json_data=item)#, convert_lists=True)
                ele['chunking_status'] = Response_Status_Messages.SUCCESS.value
            else:
                ele['chunking_status'] = Response_Status_Messages.FAILURE.value
                ele['chunking_error'] = Error_Messages.INVALID_INPUT

            all_chunks += [ele]

        return all_chunks


    def generate_embeddings(self,
        docs: list[str] | str,
        model_provider: str = 'huggingface',
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        model_dimension: PositiveInt = 768,
        model_provider_kwargs: dict[str, Any] = {
            'encode_kwargs': {"normalize_embeddings": True}
        }
    ) -> list[list[PositiveFloat]]:
        """
        Generates embeddings for the given string or list of strings using specified model.

        Args:
            docs (list[str] | str): A string or list of strings to generate embeddings for.
            model_provider (str): Model provider that provides embedding model. Default is 'huggingface'.
            model_name (str): Model to use for generating embeddings. Default is 'sentence-transformers/all-mpnet-base-v2'.
            model_dimension (int): Dimension of the model. Default is 768.
            model_provider_kwargs (dict[str, Any]): Additional configuration to pass to the model provider while creating the model instance. Default is {'encode_kwargs': {"normalize_embeddings": True}}).
        Returns:
            list[list[PositiveFloat]]: List of list of embeddings for the given strings.
        """
        from .embedder import _Embedder

        embedder = _Embedder(
            model_provider=model_provider,
            model_name=model_name,
            model_dimension=model_dimension,
            model_provider_kwargs=model_provider_kwargs
        )
        embeddings = embedder.generate_embeddings(docs=docs)

        return embeddings


    def create_bulk_objects(self,
        docs: list[dict[str, Any]],
        no_of_docs_per_bulk_object: PositiveInt = 5
    ) -> list[str]:
        """
        Creates bulk objects for OpenSearch bulk API. Each object contains list of docs on which given action (create, update, delete) will be performed.

        Args:
            docs (list[dict[str, Any]]): List of documents to be included in the bulk objects. Each document should have the structure mentioned below.
            The input docs should be in the following format:
            {
                'action': 'create'|'update'|'delete',
                'index': 'index_name',
                'doc_id': 'document_id', # mandatory for update and delete operations
                'data': { ... } # actual doc required for create and update operations
            }
            no_of_docs_per_bulk_object (int): Number of documents to be included in each bulk object. Default is 5.
        Returns:
            list[str]: List of bulk objects containing multiple docs in string format that can be directly passed to OpenSearch bulk API.
        """
        bulk_docs_body = ''
        bulk_objects = []

        i=no_of_docs_per_bulk_object

        for doc in docs:
            action_metadata = {
                doc['action']: {
                    '_index': doc['index'],
                    '_id': doc['doc_id']
                }
            }
            data = doc['data']
            bulk_docs_body += json.dumps(action_metadata) + '\n' + json.dumps(data) + '\n'

            i -= 1
            if not i:
                bulk_objects += [bulk_docs_body]

                bulk_docs_body = ''
                i=no_of_docs_per_bulk_object

        if i:
            bulk_objects += [bulk_docs_body]

        return bulk_objects



    def similaity_search(self,
        opensearch_client: OpenSearch,
        index: str,
        embeddings_field_name: str,
        query_embeddings: list[float],
        k: int = 10,
        top: int = 10,
        query_object_attributes: Optional[dict] = None,
        source_object_attributes: Optional[dict] = None,
    ):
        """
        Performs similarity search on the given OpenSearch index using the provided query embeddings and returns the most similar documents.

        Args:
            opensearch_client (OpenSearch): OpenSearch client instance to perform the search.
            index (str): Name of the OpenSearch index to search.
            embeddings_field_name (str): Name of the field in the OpenSearch index where document embeddings are stored.
            query_embeddings (list[float]): Embeddings for the query string.
            k (int): Number of nearest neighbors to search for. Default is 10.
            top (int): Number of top similar documents to return. Default is 10.
            query_object_attributes (Optional[dict]): Additional specifications for the query to be added in query object.
            source_object_attributes (Optional[dict]): Additional specifications for the source object.
        Returns:
            dict: Search results returned by OpenSearch.
        """
        query = {
            "size": top,      # returns top 'size' docs having highest _score, from below "query" result
            **(source_object_attributes or {}),
            "query": {
                **(query_object_attributes or {}),
                "knn": {
                    embeddings_field_name: {
                        "vector": query_embeddings,
                        "k": k     # searches top k nearest neighbors (most similar docs)
                    }
                }
            }
        }
        response = opensearch_client.search(
            body = query,
            index = index
        )
        return response


    def combine_chunked_docs(self,
        string_chunks: list[dict[str, Any]] = [{}],
        json_chunks: list[dict[str, Any]] = [{}],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Combines chunked documents into a format suitable for understanding.
        It takes two types of chunks, one string type and another json type, and combines each list of chunks into a single document.

        Args:
            string_chunks (list[dict[str, Any]]): List of chunks of string type. Each list of dict should have the structure mentioned below.
            [
                {
                    'chunks_type': str,
                    'chunks': ['','',],
                    'metadata': {},
                },
            ]
            json_chunks (list[dict[str, Any]]): List of chunks of json type. Each list of dict should have the structure mentioned below.
            [
                {
                    'chunks_type': [list|dict|tuple]
                    'chunks': ['','',]
                    'metadata': {},
                },
            ]
        Returns:
            dict: A dictionary containing two keys 'string_chunks' and 'json_chunks' with their respective combined chunks.
        """
        from langchain_core.documents import Document

        for doc in string_chunks:
            doc['combined_chunks'] = [
                Document(
                    page_content = chunk,
                    metadata = {
                        "source": "text",
                        "type": "unstructured",
                        **doc.get('metadata', {})
                    }
                ) for chunk in doc['chunks']
            ]


        for doc in json_chunks:
            doc['combined_chunks'] = [
                Document(
                page_content = json.dumps(chunk, ensure_ascii=False),
                    metadata = {
                        "source": "json",
                        "type": "structured",
                        **doc.get('metadata', {})
                    }
                ) for chunk in doc['chunks']
            ]


        return {
            'string_chunks': string_chunks,
            'json_chunks': json_chunks,
        }


    def create_prompt_template(self,
        prompt: str,
        data_for_prompt: Optional[dict[str, str]] = None
    ) -> ChatPromptTemplate:
        """
        Creates langchain prompt template using the given prompt and data for prompt.

        Args:
            prompt (str): The data to be passed in the prompt template object.
            data_for_prompt (Optional[dict[str, str]]): Data to pass in the prompt.
        Returns:
            ChatPromptTemplate: Prompt template object of type ChatPromptTemplate library of langchain.
        """
        human_ip = ""
        for k,v in (data_for_prompt or {}).items():
            human_ip += f"{k}: {v}\n"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),         # role = 'system'
            ("human", human_ip[:-2])    # role = 'human'
        ])

        return prompt_template


    def generate_answer(self,
        prompt_template: Any,
        llm: Any,
        data_for_prompt: Optional[dict[str, str]] = None
    ) -> str:
        """
        Generates answer by invoking the chain of prompt template and provided llm.

        Args:
            prompt_template (Any): The prompt template object of langchain prompt template library containing the prompt.
            llm (Any): The language model used to generate answer for the prompt.
            data_for_prompt (Optional[dict[str, str]]): Data to pass in the chain invocation.
        Returns:
            str: The generated answer from the llm.
        """
        answer_chain = prompt_template | llm
        response = answer_chain.invoke(data_for_prompt or {})

        return response


