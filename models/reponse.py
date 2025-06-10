from pydantic import BaseModel

class info_response(BaseModel):
    embedding_model : str
    generation_model : str
    vector_store_path : str
    vector_store_index : str

class ragit_response(BaseModel):
    id: str
    user_query: str
    rag_result: str
    model_used: str