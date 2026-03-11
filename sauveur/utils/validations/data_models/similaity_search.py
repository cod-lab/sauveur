from pydantic import BaseModel, ConfigDict#, PositiveInt
from typing import Optional
from opensearchpy import OpenSearch

from sauveur.configs.default_env_vars import Default_Env_Vars


class Source_Object_Model(BaseModel):
    _source: dict[str, dict]


class Similaity_Search_Model(BaseModel):
    opensearch_client: OpenSearch
    index: str
    embeddings_field_name: str
    query_embeddings: list[float]
    k: int = Default_Env_Vars.K
    top: int = Default_Env_Vars.TOP
    query_object_attributes: Optional[dict[str, dict]]
    source_object_attributes: Optional[Source_Object_Model]


    model_config = ConfigDict(arbitrary_types_allowed=True)     # it allows external datatypes, here we are using 'OpenSearch' datatype






