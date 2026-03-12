from typing import Any, Optional
import json
from pprint import pprint as pp

from pydantic import PositiveInt, PositiveFloat
from opensearchpy import OpenSearch
from langchain_core.prompts import ChatPromptTemplate


from sauveur.helpers.errors_messages import Error_Messages
from sauveur.helpers.reponse_status_messages import Response_Status_Messages


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
                ele['chunking_status'] = Response_Status_Messages.SUCCESS
            elif isinstance(item, (list, dict)):
                ele['chunks'] = json_splitter.split_json(json_data=item)#, convert_lists=True)
                ele['chunking_status'] = Response_Status_Messages.SUCCESS
            else:
                ele['chunking_status'] = Response_Status_Messages.FAILURE
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
        """
        from sauveur.embedder import _Embedder

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
        with each doc there should be an index and operation associated and doc_id is mandatory for update and delete opt
        doc = [
            {
                'action': 'create',
                'index': 'index1',
                'doc_id': 'id1',
                'data': {},
            },
            {
                'action': 'create',
                'index': 'index1',
                'doc_id': 'id1',
                'data': {},
            },

        ]
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
        string_chunks: [
            {
                'chunks_type': str,
                'chunks': ['','',],
                'metadata': {},
            },
        ]
        json_chunks: [
            {
                'chunks_type': [list|dict|tuple]
                'chunks': ['','',]
                'metadata': {},
            },
        ]
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
        """
        answer_chain = prompt_template | llm
        response = answer_chain.invoke(data_for_prompt or {})

        return response




