from pydantic import BaseModel#, ConfigDict#, PositiveInt
from typing import Any

from sauveur.configs.env_vars import Env_Vars


class String_Chunks_Model(BaseModel):
    chunks_type: str = Env_Vars.STRING_CHUNKS_TYPE
    chunks: list[str]
    metadata: dict[str, Any]

class Json_Chunks_Model(BaseModel):
    chunks_type: str = Env_Vars.JSON_CHUNKS_TYPE
    chunks: list[str]
    metadata: dict[str, Any]


class Combine_Chunked_Docs_Model(BaseModel):
    string_chunks: list[String_Chunks_Model]
    json_chunks: list[Json_Chunks_Model]



