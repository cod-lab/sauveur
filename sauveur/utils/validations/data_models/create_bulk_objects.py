from pydantic import BaseModel, JsonValue, PositiveInt

from sauveur.configs.default_env_vars import Default_Env_Vars


class Docs(BaseModel):
    action: str
    index: str
    doc_id: str
    data: JsonValue


class Create_Bulk_Objects_Model(BaseModel):
    docs: list[Docs]
    no_of_docs_per_bulk_object: PositiveInt = Default_Env_Vars.NO_OF_DOCS_PER_BULK_OBJECT


