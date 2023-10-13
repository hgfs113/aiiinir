from fastapi import FastAPI
from pydantic import BaseModel
from app.chat import CHATGPT

app = FastAPI()

chat = CHATGPT()

class UserText(BaseModel):
    message: str
    user_id: str

@app.post("/message")
async def create_item(item: UserText):
    return {"message": item.message, "result": chat(item.message)['result'], "user_id": item.user_id}

