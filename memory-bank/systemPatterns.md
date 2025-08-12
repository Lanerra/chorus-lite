# **System Patterns: Chorus-Lite**

## **Core Architecture Patterns**

1. **Multi-Agent Societal Model**
   - Specialized agents (Story Architect, Scene Generator, Integration Manager) operate as a decentralized society
   - Agents communicate through a centralized orchestrator (LangGraph-based)
   - Each agent maintains its own responsibility domain with strict boundaries
   - *Pattern Implementation*: `src/chorus/langgraph/orchestrator.py` and `src/chorus/agents/`

2. **Stateful Narrative Coherence**
   - Canonical database (`src/chorus/canon/`) tracks all story elements
   - All agents reference the same persistent state for consistency
   - *Pattern Implementation*: `src/chorus/canon/db.py` and `src/chorus/canon/models.py`

3. **LangGraph Workflow Orchestration**
   - Stateful workflow management with error recovery
   - Graph nodes represent agent steps with explicit transitions
   - *Pattern Implementation*: `src/chorus/langgraph/graph.py` and `src/chorus/langgraph/error_recovery.py`

## **Key Technical Decisions**

- **Database-First Approach**: All story elements stored in PostgreSQL via SQLAlchemy
- **Agent Specialization**: Each agent has a single responsibility (no "jack-of-all-trades" agents)
- **State Persistence**: Complete story state is saved after each agent step
- **Web Interface Integration**: Real-time updates via WebSocket (`src/chorus/web/websocket.py`)

## **Critical Implementation Paths**

1. **Story Development Flow**:
   ```
   User Concept → Story Architect (Outline) → Scene Generator (Scenes) → Integration Manager (Merge) → Final Story
   ```

2. **Coherence Enforcement**:
   - All agents query canonical database before generating content
   - Changes to entities trigger coherence validation
   - Violations are handled by error recovery system

3. **State Management**:
   - Each agent step updates story state in database
   - Complete state preserved for resumable workflows
   - Versioning ensures historical context can be restored
