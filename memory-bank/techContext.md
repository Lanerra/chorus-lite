# Chorus-Lite Technical Context

## Core Technologies
- **Python 3.12+**: Primary development language
- **PostgreSQL 1+**: Primary database with pgvector extension
- **FastAPI**: Web framework for API and web interface
- **LangGraph**: Workflow orchestration framework
- **Poetry**: Dependency management
- **Docker**: For local PostgreSQL setup
- **Ruff**: Code formatting and linting
- **Mypy**: Static type checking

## Development Setup
```bash
# Environment setup
cp .env.example .env
# Edit .env with appropriate values
poetry install
poetry shell
```

## Configuration Management
Configuration uses Pydantic models with environment variables:
```python
class ChorusConfig(BaseModel):
    database: DatabaseConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    agents: AgentModelConfig
    system: SystemConfig
```

## Key Technical Constraints
1. **Async-First Architecture**: All I/O operations use async/await
2. **Structured Output**: Pydantic models for data validation throughout
3. **State Management**: LangGraph workflow state managed through TypedDict
4. **Database Schema**: PostgreSQL with custom enum types and UUID primary keys
5. **Error Handling**: Comprehensive error recovery strategies (retry, skip, rollback)

## Memory Management
- **In-Memory Store**: For scene memory and context
- **Text Embeddings**: For similarity search and memory retrieval
- **Caching Strategy**: 
  - World Generation: 1-hour TTL
  - Character Profiles: 30-minute TTL
  - Scene Memory: In-memory storage

## Performance Considerations
- **Task Concurrency**: Configurable scene processing concurrency
- **Circuit Breakers**: Prevent resource exhaustion
- **Rate Limiting**: Token bucket algorithm for event processing
- **Connection Pooling**: Efficient database connection reuse

## Security Measures
- **API Key Management**: Environment variables for LLM API keys
- **Input Validation**: Pydantic model validation for all inputs
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Sanitized user input
- **HTTPS Support**: TLS encryption for web interface

## Testing Strategy
- **Unit Tests**: Individual component testing in `tests/`
- **Integration Tests**: Multi-component workflow validation
- **Performance Tests**: Load and stress testing
- **Web Tests**: UI and API endpoint validation

## Development Workflow
1. **Feature Branches**: Isolated development work
2. **Conventional Commits**: Standardized commit messages
3. **Pull Requests**: Code review process
4. **Squash Merging**: Clean commit history
5. **CI Pipeline**: Automated testing and quality checks

## Tooling
- **Ruff**: For code formatting (`poetry run ruff check src`)
- **Mypy**: For type checking (`poetry run mypy src`)
- **Poetry**: Dependency management
- **Alembic**: Database migrations
- **Docker**: Local PostgreSQL setup for development

## Configuration Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost:5432/chorus` |
| `OPENAI_API_BASE` | LLM API endpoint | `https://api.openai.com/v1` |
| `STORY_ARCHITECT_MODEL` | Story architect LLM model | `gpt-4-turbo` |
| `SCENE_GENERATOR_MODEL` | Scene generator LLM model | `gpt-4-turbo` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `TEMPERATURE` | LLM sampling temperature | `0.7` |

## Critical Dependencies
- **pgvector**: For vector search and embeddings
- **LangGraph**: For workflow orchestration
- **FastAPI**: For web interface and API
- **Pydantic**: For data validation
- **SQLAlchemy**: For database operations

## Error Handling Strategy
- **Retry with Exponential Backoff**: Automatic retries for transient errors
- **Circuit Breaker Pattern**: Prevent cascading failures
- **Fault Isolation**: Isolate failing components
- **State Rollback**: Restore to previous valid state
- **Manual Intervention**: Escalate to human review for critical errors
