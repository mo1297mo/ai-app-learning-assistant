from pydantic import BaseModel



class UserQuery(BaseModel):
    question: str
    model_choice: str

