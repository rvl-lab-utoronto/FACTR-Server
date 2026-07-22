import asyncio
import json
from websockets.asyncio.client import connect


async def hello():
    async with connect("ws://localhost:5000/ws/left") as websocket:
        while True:
            message = await websocket.recv()
            frame = json.loads(message)
            print(frame["type"], frame)


if __name__ == "__main__":
    asyncio.run(hello())
