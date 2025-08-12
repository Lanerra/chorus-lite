# src/chorus/web/main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Create the FastAPI application
app = FastAPI(
    title="Chorus-Lite Web Interface",
    description="Web interface for Chorus-Lite story generation system",
)

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="src/chorus/web/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="src/chorus/web/templates")

# Include routes
from src.chorus.web.routes import router

app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
