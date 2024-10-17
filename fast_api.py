from fastapi import FastAPI
from pydantic import BaseModel
from src.server.checkin import Checking_Engine
from typing import Any


class Checkin_Item(BaseModel):
    user_name: str
    upload_image: Any


app = FastAPI()
eng = Checking_Engine()

@app.post('/checkin')
def index(inputs : Checkin_Item):
    print("Server side values: ",inputs.user_name)

if __name__ == '__main__':
    app.run