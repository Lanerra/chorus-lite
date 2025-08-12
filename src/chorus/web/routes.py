# src/chorus/web/routes.py
import asyncio
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from src.chorus.agents.story_architect import StoryArchitect
from src.chorus.canon.crud import get_all_chapters, get_all_characters, get_all_scenes
from src.chorus.canon import get_pg  # Add import for get_pg
from src.chorus.core.env import get_settings
from src.chorus.langgraph.graph import build_graph
from src.chorus.web.websocket import websocket_manager  # Import the WebSocket manager
from chorus.core.logging import get_logger

logger = get_logger(__name__)

# Create the router
router = APIRouter()

# Initialize the story architect agent
story_architect = StoryArchitect(model=get_settings().agents.story_architect)

# Real story ideas from database
@router.get("/api/story-ideas")
async def get_story_ideas():
    """Get all story ideas from the database."""
    try:
        async with get_pg() as session:
            # Assuming there's a StoryIdea model in the database
            # If not, we'll need to create one
            from sqlalchemy import select
            from src.chorus.models.story import StoryIdea
            story_ideas = await session.execute(select(StoryIdea.__table__))
            ideas = story_ideas.scalars().all()

            return {
                "story_ideas": [
                    {
                        "id": str(idea.id),
                        "title": idea.title,
                        "description": idea.description,
                        "created_at": idea.created_at,
                        "updated_at": idea.updated_at,
                    }
                    for idea in ideas
                ]
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving story ideas: {str(e)}"
        )


@router.get("/api/database")
async def get_database_content():
    """Get all database content organized by category."""
    try:
        # Get characters
        async with get_pg() as session:
            characters = await get_all_characters(session)

        # Get chapters
        async with get_pg() as session:
            chapters = await get_all_chapters(session)

        # Get scenes
        async with get_pg() as session:
            scenes = await get_all_scenes(session)

        # Format the data for display
        formatted_data = {
            "characters": [
                {
                    "id": str(char.id),
                    "name": char.name,
                    "full_name": char.full_name,
                    "aliases": char.aliases,
                    "age": char.age,
                    "birth_date": char.birth_date,
                    "death_date": char.death_date,
                    "species": char.species,
                    "role": char.role,
                    "rank": char.rank,
                    "backstory": char.backstory,
                    "beliefs": char.beliefs,
                    "desires": char.desires,
                    "intentions": char.intentions,
                    "motivations": char.motivations,
                    "fatal_flaw": char.fatal_flaw,
                    "arc": char.arc,
                    "voice": char.voice,
                }
                for char in characters
            ],
            "chapters": [
                {
                    "id": str(chap.id),
                    "title": chap.title,
                    "description": chap.description,
                    "order_index": chap.order_index,
                    "structure_notes": chap.structure_notes,
                    "scene_count": len(chap.scenes) if chap.scenes else 0,
                }
                for chap in chapters
            ],
            "scenes": [
                {
                    "id": str(scene.id),
                    "title": scene.title,
                    "text": scene.text,
                    "status": scene.status,
                    "scene_number": scene.scene_number,
                    "character_count": len(scene.characters) if scene.characters else 0,
                    "created_at": scene.created_at,
                    "updated_at": scene.updated_at,
                }
                for scene in scenes
            ],
        }

        return formatted_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving database content: {str(e)}"
        )


@router.post("/api/generate")
async def generate_story(idea: dict):
    # Validate input
    if not idea or "idea" not in idea:
        raise HTTPException(status_code=400, detail="Story idea is required")

    story_idea = idea["idea"]

    # Generate a unique story ID
    story_id = str(uuid4())

    # Create a new story session
    try:
        # Initialize the graph with the story idea
        from src.chorus.langgraph.checkpointer import get_checkpointer  # Import checkpointer
        checkpointer = get_checkpointer()
        graph = build_graph(checkpointer=checkpointer)  # Pass the checkpointer

        # Start the story generation process through the orchestrator
        # This is the key change - we're now actually running the graph
        from src.chorus.langgraph.orchestrator import create_story
        thread_id, final_state = await create_story(story_idea, session_id=story_id)

        # Send completion message
        await websocket_manager.broadcast(
            {
                "type": "log",
                "message": "Story generation completed successfully!",
                "level": "success",
            }
        )

    except Exception as e:
        # Handle any errors during generation
        await websocket_manager.broadcast(
            {
                "type": "log",
                "message": f"Error during story generation: {str(e)}",
                "level": "error",
            }
        )
        # Log the error with more detail
        logger.error(f"Error in generate_story: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

    # Return success response
    return {
        "story_id": story_id,
        "status": "generation_completed",
        "message": "Story generation started successfully",
    }


@router.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Stream actual logs from the story generation process
        # Logs are broadcasted via websocket_manager during story generation
        while True:
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
