
# -- FastAPI --
from fastapi import FastAPI

class FACTRAPI():
    def __init__(self):
        self.app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Hello World"}