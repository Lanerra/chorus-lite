# Chorus-Lite Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Agent Architecture](#agent-architecture)
5. [LangGraph Workflow](#langgraph-workflow)
6. [Database Schema](#database-schema)
7. [Web Interface](#web-interface)
8. [Configuration](#configuration)
9. [Error Handling](#error-handling)
10. [Deployment](#deployment)

## Overview

Chorus-Lite is an autonomous, agentic, creative-writing system designed to generate long-form fiction through a streamlined three-agent architecture. Built on LangGraph, it leverages PostgreSQL for canonical storage and provides a web interface for monitoring and interaction.

### Key Features
- **Autonomous Operation**: Runs with minimal human intervention
- **Agentic Architecture**: Multi-agent system for specialized writing tasks
- **LangGraph Orchestration**: Workflow management and state tracking
- **PostgreSQL Storage**: Canonical database for persistent story data
- **Real-time Monitoring**: WebSocket-based live updates
- **Structured Output**: Pydantic models for data validation

## System Architecture

The Chorus-Lite system follows a streamlined architecture focused on essential functionality:

### Simplified Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │◄──►│   FastAPI Server │◄──►│   PostgreSQL    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   LangGraph      │
                    │   Workflow       │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Story       │      │ Scene       │      │ Integration │
│ Architect   │      │ Generator   │      │ Manager     │
└─────────────┘      ┌─────────────┘      └─────────────┘
                     └─────────────┘
```

### Component Layers
1. **Interface Layer**: Web UI and API endpoints
2. **Orchestration Layer**: LangGraph workflow management
3. **Agent Layer**: Three core agents (StoryArchitect, SceneGenerator, IntegrationManager) handling all story creation tasks
4. **Persistence Layer**: PostgreSQL database for canonical storage
5. **Core Layer**: Shared utilities and infrastructure

## Core Components

### Core Utilities (`src/chorus/core/`)
- **`llm.py`**: LiteLLM wrapper for LLM calls with structured output support
- **`logs.py`**: Enhanced structured logging with WebSocket streaming
- **`env.py`**: Environment configuration loading
- **`embedding.py`**: Text embedding utilities
- **`queue.py`**: Task queue management
- **`cache.py`**: Caching utilities
- **`communication.py`**: Inter-agent communication
- **`name_utils.py`**: Name processing utilities
- **`output_utils.py`**: Debug output and snapshot utilities

### Configuration (`src/chorus/config/`)
The system uses Pydantic models for configuration management:

```python
class ChorusConfig(BaseModel):
    database: DatabaseConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    agents: AgentModelConfig
    system: SystemConfig
    # ... other configurations
```

### Models (`src/chorus/models/`)
Pydantic models for data validation and serialization:
- **Story Models**: `Story`, `Chapter`, `Scene`, `SceneStatus`
- **Character Models**: `CharacterProfile`, `CharacterRelationship`
- **World Models**: `WorldAnvil`, `Concept`
- **Task Models**: `SceneBrief`, `RewriteTask`, `ReviseTask`
- **Feedback Models**: Feedback models are deprecated in Chorus-Lite as their functionality is now handled directly by the core agents

## Agent Architecture

### Base Agent (`src/chorus/agents/base.py`)
All agents inherit from the `Agent` base class which provides:
- LLM calling with error handling
- Structured output validation
- Retry mechanisms
- Database connection management
- Task queue operations
- Scene status management

### Core Agents

The Chorus-Lite system has been simplified to focus on three core agents that handle all aspects of story creation:

#### StoryArchitect (`src/chorus/agents/story_architect.py`)

Responsible for high-level story planning and world-building:
- **Concept Generation**: Creates multiple story concepts from ideas
- **Story Planning**: Generates comprehensive story outlines
- **Character Creation**: Creates detailed character profiles
- **World Building**: Generates world elements and lore

Key methods:
- `generate_concepts()`: Generate story concept options
- `create_story_outline()`: Create high-level story structure
- `generate_profiles()`: Create character profiles
- `generate_world_anvil()`: Generate world elements

#### SceneGenerator (`src/chorus/agents/scene_generator.py`)

Handles scene writing and revision:
- **Scene Writing**: Generates scene content from briefs
- **Scene Revision**: Enhances scenes based on feedback and style guidelines

Key methods:
- `write_scene()`: Generate scene content
- `generate_complete_scene()`: Full scene generation pipeline

#### IntegrationManager (`src/chorus/agents/integration_manager.py`)

Manages story integration and finalization:
- **Scene Integration**: Combines scenes into chapters
- **Story Validation**: Ensures completeness and consistency across the narrative arc
- **Publication Preparation**: Creates export formats for completed stories

Key methods:
- `integrate_scenes()`: Combine scenes into chapters
- `validate_story_structure()`: Check story completeness
- `finalize_story()`: Prepare final story for export
- `create_publication_package()`: Generate publication-ready formats

## LangGraph Workflow

### Graph Structure (`src/chorus/langgraph/graph.py`)
The workflow is split into modular subgraphs:

#### Story Setup Subgraph
1. `generate_concepts` → `select_concept` → `generate_world`
2. `generate_profiles` → `seed_narrative_context` → `generate_outline`
3. `prepare_scenes` → `dequeue_scene`

#### Revision Loop Subgraph
1. `lore_context` → `retrieve_memory` → `draft_scene`
2. `coherence_check` → `finalize_scene` or `revise_scene`
3. `store_memory` → `catalog_lore` → `evolve_profiles`
4. `summarize_memory` → `dequeue_scene` (loop)

### State Management (`src/chorus/langgraph/state.py`)
Uses `StoryState` TypedDict for workflow state:
```python
class StoryState(TypedDict, total=False):
    vision: Concept
    world_info: list[WorldAnvil]
    character_profiles: list[CharacterProfile]
    current_scene_id: str
    scene_briefs: list[SceneBrief]
    # ... many other fields
```

### Error Recovery (`src/chorus/langgraph/error_recovery.py`)
Comprehensive error handling with multiple recovery strategies:
- **Retry**: Exponential backoff with circuit breaker
- **Skip**: Bypass non-critical failures
- **Rollback**: Restore previous valid state
- **Circuit Break**: Isolate failing components
- **Manual Intervention**: Escalate to human review

### Memory Management (`src/chorus/langgraph/memory.py`)
In-memory store for long-term context:
- Scene memory storage and retrieval
- Text embedding for similarity search
- Memory summarization utilities

## Database Schema

### Core Tables

#### `scene` Table
```sql
CREATE TABLE scene (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    text TEXT,
    status scene_status_enum NOT NULL DEFAULT 'queued',
    scene_number INTEGER,
    setting TEXT,
    characters UUID[],
    location_id UUID,
    chapter_id UUID REFERENCES chapter(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `character_profile` Table
```sql
CREATE TABLE character_profile (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    full_name TEXT,
    aliases TEXT[],
    gender TEXT,
    age INTEGER,
    birth_date DATE,
    death_date DATE,
    species TEXT,
    role TEXT,
    rank TEXT,
    backstory TEXT,
    beliefs TEXT[],
    desires TEXT[],
    intentions TEXT[],
    motivations TEXT[],
    fatal_flaw TEXT,
    arc TEXT,
    voice TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `chapter` Table
```sql
CREATE TABLE chapter (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    order_index INTEGER NOT NULL,
    structure_notes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `world_anvil` Table
```sql
CREATE TABLE world_anvil (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    category TEXT,
    tags TEXT[],
    location_type TEXT,
    ruling_power TEXT,
    cultural_notes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Enum Types
```sql
CREATE TYPE scene_status_enum AS ENUM (
    'queued',
    'drafting',
    'in_review',
    'rejected',
    'approved'
);
```

### CRUD Operations (`src/chorus/canon/crud.py`)
High-level database operations:
- Character profile management
- Scene creation and updating
- Chapter management
- World element storage
- Feedback recording
- Style guide operations

## Web Interface

### FastAPI Server (`src/chorus/web/`)
Built with FastAPI providing:
- Static file serving for web assets
- REST API endpoints
- WebSocket connections for real-time updates
- Jinja2 templates for HTML rendering

### API Endpoints
- `GET /`: Web interface
- `GET /api/story-ideas`: Sample story ideas
- `GET /api/database`: Database content retrieval
- `POST /api/generate`: Story generation initiation
- `GET /health`: Health check endpoint

### WebSocket Communication (`src/chorus/web/websocket.py`)
Real-time event streaming:
- Live log updates
- Status notifications
- Database change notifications
- Error reporting

### Frontend (`src/chorus/web/static/`)
Modern web interface built with vanilla JavaScript:
- **HTML Template** (`templates/index.html`): Main application layout
- **CSS Styles** (`static/css/style.css`): Responsive design
- **JavaScript** (`static/js/app.js`): Client-side logic and WebSocket handling

Key features:
- Real-time log streaming
- Database content display
- Story generation form
- Status monitoring
- Responsive design

## Configuration

### Environment Variables
Key configuration variables:
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_BASE`: LLM API endpoint
- `OPENAI_API_KEY`: LLM API key
- `STORY_ARCHITECT_MODEL`: Story architect LLM model
- `SCENE_GENERATOR_MODEL`: Scene generator LLM model
- `INTEGRATION_MANAGER_MODEL`: Integration manager LLM model
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `TEMPERATURE`: LLM sampling temperature

### Configuration Models (`src/chorus/config/config.py`)
Structured configuration using Pydantic:
```python
class ChorusConfig(BaseModel):
    database: DatabaseConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    agents: AgentModelConfig
    system: SystemConfig
    cache: CacheConfig
    retry: RetryConfig
    # ... additional configuration sections
```

## Error Handling

### Structured Logging (`src/chorus/core/logs.py`)
Comprehensive event logging system:
- **Event Types**: System, workflow, scene generation, user actions
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Priority System**: Critical, High, Normal, Low for rate limiting
- **WebSocket Streaming**: Real-time event delivery
- **Metrics Tracking**: Performance monitoring and statistics

### Error Recovery Strategies
1. **Retry with Exponential Backoff**: Automatic retry with increasing delays
2. **Circuit Breaker Pattern**: Prevent cascading failures
3. **Fault Isolation**: Isolate failing components
4. **State Rollback**: Restore to previous valid state
5. **Skip Non-Critical Operations**: Continue with degraded functionality
6. **Manual Intervention Escalation**: Human review for complex errors

### Error Classification
Errors are classified into categories:
- **Transient**: Temporary failures (network timeouts, rate limits)
- **Permanent**: Configuration errors, data corruption
- **Validation**: Input validation failures
- **Resource**: Memory, disk space, connection limits
- **Business Logic**: Workflow constraint violations

## Deployment

### Prerequisites
- Python 3.12+
- PostgreSQL 16+
- Poetry for dependency management
- Docker (optional, for PostgreSQL)

### Setup Process
1. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Edit .env with appropriate values
   ```

2. **Database Initialization**:
   ```bash
   docker compose up -d  # Start PostgreSQL
   poetry install        # Install dependencies
   poetry shell          # Activate virtual environment
   ```

3. **Database Migration**:
   ```bash
   alembic upgrade head  # Apply database schema
   ```

4. **Application Start**:
   ```bash
   poetry run chorus      # Start web interface
   ```

### Quality Assurance
Before committing changes:
- Run code formatting: `poetry run ruff check src`
- Run type checking: `poetry run mypy src`
- Run tests: `poetry run pytest`

### Testing Framework
Comprehensive test suite:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **Performance Tests**: Load and stress testing
- **Web Tests**: UI and API endpoint validation

Test structure follows source code organization in `tests/` directory.

## Development Workflow

### Code Standards
- **Async-First**: All I/O operations use async/await
- **Pydantic Models**: Strong typing with validation
- **Google-Style Docstrings**: Consistent documentation
- **Ruff Formatting**: Automatic code formatting
- **Mypy Type Checking**: Static type analysis

### Git Workflow
- **Conventional Commits**: Standardized commit messages
- **Feature Branches**: Isolated development work
- **Pull Requests**: Code review process
- **Squash Merging**: Clean commit history

### Continuous Integration
- Automated testing on all commits
- Code quality checks
- Security scanning
- Deployment validation

## Performance Considerations

### Caching Strategy
- **World Generation Cache**: 1-hour TTL for expensive operations
- **Character Profile Cache**: 30-minute TTL for frequently accessed data
- **Scene Memory Cache**: In-memory storage for context

### Task Management
- **Scene Processing**: Configurable concurrency levels for scene generation
- **Circuit Breakers**: Prevent resource exhaustion
- **Rate Limiting**: Token bucket algorithm for event processing
- **Connection Pooling**: Efficient database connection reuse

### Memory Management
- **Session Isolation**: Prevents cross-session contamination in the three-agent workflow
- **State Cleanup**: Automatic cleanup of stale data from completed story phases
- **Resource Constraints**: Configurable limits for agent operations

## Security Considerations

### Authentication
- **API Key Protection**: LLM API key management
- **Web Token**: Optional web interface authentication
- **Database Security**: PostgreSQL authentication and authorization

### Data Protection
- **Environment Variables**: Sensitive data in .env files
- **Input Validation**: Pydantic model validation
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Sanitized user input

### Network Security
- **HTTPS Support**: TLS encryption for web interface
- **WebSocket Security**: Secure real-time communication
- **CORS Configuration**: Controlled cross-origin requests

## Monitoring and Observability

### Logging System
- **Structured Events**: JSON-formatted log entries
- **Real-time Streaming**: WebSocket-based log delivery
- **Performance Metrics**: Latency and throughput tracking
- **Error Tracking**: Comprehensive error reporting

### Health Checks
- **System Health**: Overall system status
- **Component Health**: Individual service status
- **Database Health**: Connection and query performance
- **LLM Health**: API connectivity and response times

### Performance Monitoring
- **Event Processing Latency**: Real-time event delivery performance
- **Database Query Performance**: SQL execution timing
- **LLM Response Times**: API call performance
- **Memory Usage**: Resource consumption tracking

## Future Enhancements

### Planned Features
- **Advanced NER Integration**: More sophisticated entity recognition
- **Enhanced Memory System**: Improved context retention
- **Multi-Language Support**: Internationalization capabilities
- **Plugin Architecture**: Extensible agent and workflow system

### Scalability Improvements
- **Distributed Processing**: Multi-node deployment
- **Load Balancing**: Horizontal scaling support
- **Caching Layers**: Redis-based caching
- **Message Queues**: Advanced task distribution

### AI/ML Enhancements
- **Fine-tuned Models**: Domain-specific LLM training
- **Reinforcement Learning**: Adaptive workflow optimization
- **Advanced Prompting**: Dynamic prompt engineering
- **Quality Scoring**: Automated content evaluation

---

*This documentation provides a comprehensive overview of the Chorus-Lite system architecture and implementation details. It serves as a reference for developers working with or extending the system.*
