from pydantic import BaseModel, PositiveInt
from typing import Any#, Optional

from sauveur.configs.default_env_vars import Default_Env_Vars


class Generate_Embeddings_Model(BaseModel):
    docs: list[str] | str
    model_provider: str = Default_Env_Vars.MODEL_PROVIDER
    model_name: str = Default_Env_Vars.MODEL_NAME
    model_dimension: PositiveInt = Default_Env_Vars.MODEL_DIMENSION
    model_provider_kwargs: dict[str, Any] = Default_Env_Vars.MODEL_PROVIDER_KWARGS


