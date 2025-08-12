# src/chorus/web/websocket.py
from fastapi import WebSocket
from typing import List, Dict, Any
import asyncio

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Send a message to all connected clients."""
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                # If there's an error sending to a client, remove them
                print(f"Error sending message to client: {e}")
                self.disconnect(connection)

# Create a global instance
websocket_manager = WebSocketManager()
