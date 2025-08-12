# Chorus-Lite Product Context

## Why This Project Exists
Chorus-Lite solves the growing need for autonomous, high-quality creative writing tools that can generate long-form fiction with minimal human intervention. Current writing assistants lack the structured workflow and persistent state management needed for cohesive story development.

## Core Problems Solved
1. **Fragmented Story Development**: Writers struggle to maintain consistency across multiple scenes and chapters
2. **Manual Workflow Management**: Time-consuming process of coordinating different writing tasks
3. **Lack of Persistence**: Previous writing assistants don't maintain context across sessions
4. **Inconsistent Style**: Difficulty maintaining consistent character voices and narrative styles

## How It Works
Chorus-Lite provides a structured, autonomous writing workflow through:
- **Three Specialized Agents** (StoryArchitect, SceneGenerator, IntegrationManager) working in concert
- **LangGraph Orchestration** for seamless workflow management and state tracking
- **PostgreSQL with pgvector** for persistent story context and memory
- **Real-time Monitoring** via WebSocket for live feedback and adjustments

## User Experience Goals
1. **Autonomous Operation**: Minimal user input required beyond initial story concept
2. **Structured Output**: Clear, consistent narrative with maintained character voices
3. **Real-time Feedback**: Live progress monitoring during story generation
4. **Iterative Refinement**: Easy scene and chapter revisions without restarting the process

## Technical Capabilities Supporting Product Goals
- **Three-Agent Specialization**: Each agent handles a distinct aspect of story creation
- **LangGraph Workflow**: Ensures logical progression from concept to completed story
- **Memory Management**: Retains context across multiple writing sessions
- **Structured Data Models**: Guarantees consistent data flow between agents
- **Error Recovery**: Maintains workflow continuity during LLM or database issues

## Technical Limitations to Be Aware Of
1. **LLM Dependency**: Quality depends on underlying LLM capabilities
2. **Context Window Limits**: Story length constrained by LLM context window
3. **Database Size**: Large stories require sufficient database storage
4. **Model Selection**: Different LLM models affect writing style and quality
