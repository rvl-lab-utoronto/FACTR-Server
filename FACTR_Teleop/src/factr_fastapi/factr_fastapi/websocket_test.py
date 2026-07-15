import asyncio
from websockets.asyncio.client import connect
import time


async def hello():
    async with connect("ws://localhost:5001") as websocket:
        while (True):
            message = await websocket.recv()
            print(message)


if __name__ == "__main__":
    asyncio.run(hello())