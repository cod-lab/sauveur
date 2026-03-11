from pydantic import BaseModel
from typing import Optional, Any


class Generate_Answer_Model(BaseModel):
    prompt_template: Any
    llm: Any
    data_for_prompt: Optional[dict[str, str]]



