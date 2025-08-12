# **Tech Context: Chorus-Lite**

## **Technologies Used**
- **Programming Language**: Python 3.10+
- **Database**: PostgreSQL with pgvector extension (for vector search)
- **Orchestration**: LangGraph (stateful workflow management)
- **ORM**: SQLAlchemy with Alembic for database migrations
- **Web Framework**: FastAPI (based on `src/chorus/web/main.py` structure)
- **Build System**: Poetry (dependency management)
- **Containerization**: Docker (via `docker-compose.yml`)

## **Development Setup**
1. **Environment Initialization**:
   - Install dependencies: `poetry install`
   - Initialize database: `python scripts/init_db.py`
   - Start development server: `uvicorn src/chorus.web.main:app --reload`

2. **Key Configuration Files**:
   - `pyproject.toml`: Project dependencies and build settings
   - `config/agents.yaml`: Agent configuration parameters
   - `alembic.ini`: Database migration configuration
   - `.env.example`: Environment variable template

## **Technical Constraints**
- **Database Schema**: Must support PostgreSQL extensions (pgvector)
- **Agent Communication**: Strictly via LangGraph state transitions
- **State Persistence**: Full story state stored in database after each agent step
- **Web Interface**: Real-time updates via WebSocket (`src/chorus/web/websocket.py`)

## **Dependencies**
- Core: `langgraph`, `sqlalchemy`, `psycopg2-binary`, `uvicorn`, `fastapi`
- Database: `pgvector`, `alembic`
- Utilities: `python-dotenv`, `python-jose`, `pydantic-settings`

## **Tool Usage Patterns**
- **Database Migrations**: `alembic upgrade head` (after schema changes)
- **Development Workflow**: 
  ```bash
  poetry shell
  uvicorn src/chorus.web.main:app --reload
  ```
- **Agent Configuration**: Modify `config/agents.yaml` for agent parameters
- **Web Interface**: Access via `http://localhost:8000` (default FastAPI port)

## **Critical Integration Points**
1. **LangGraph ↔ Canonical Database**:
   - All agents query `src/chorus/canon/` before generating content
   - Changes trigger coherence validation in `src/chorus/canon/db.py`

2. **Web Interface ↔ Agents**:
   - WebSocket updates sent via `src/chorus/web/websocket.py`
   - Real-time logs visible in browser UI
