from pydantic import BaseModel

class ragit_request(BaseModel):
    query : str
