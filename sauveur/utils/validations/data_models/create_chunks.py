from pydantic import BaseModel, PositiveInt#, JsonValue

from sauveur.configs.default_env_vars import Default_Env_Vars


class Create_Chunks_Model(BaseModel):
    items: list[
        str
        |list[str|list|tuple|dict]
        |tuple[str|list|tuple|dict]
        |dict[str,str|list|tuple|dict]
    ]
    chunk_size: PositiveInt = Default_Env_Vars.CHUNK_SIZE
    chunk_overlap: PositiveInt = Default_Env_Vars.CHUNK_OVERLAP



