from pydantic import BaseModel
from typing import Optional


class Create_Prompt_Template_Model(BaseModel):
    prompt: str
    data_for_prompt: Optional[dict[str, str]]



