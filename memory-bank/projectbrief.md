# **Project Brief: Chorus-Lite**

## **High-Level Overview**

Chorus-Lite is an autonomous, agentic, novel-writing system. It leverages a multi-agent architecture to collaboratively generate, develop, and write a complete novel from a high-level user-provided concept. The system is designed to manage the entire creative process, from initial world-building and character creation to outlining, scene generation, and final manuscript compilation.

## **Core Requirements and Goals**

The primary goal of Chorus-Lite is to produce a coherent, well-structured novel. This involves several key requirements:

* **Autonomous Operation:** The system is designed to run with minimal human intervention. It manages its own task queue and orchestrates the various agents to progress the story from concept to completion.
* **Agentic Architecture:** The system is built around a society of specialized agents, each responsible for a specific aspect of the writing process. This includes:
  * **Story Architect:** Develops the high-level plot and outline.
  * **Scene Generator:** Creates detailed prompts for individual scenes based on the outline.
  * **Integration Manager:** Reviews and integrates newly generated content into the main story.
* **Narrative Coherence:** The system must ensure that the generated story is logical, consistent, and engaging. It uses a canonical database to track characters, plot points, and relationships, preventing continuity errors.
* **Stateful and Resumable:** The entire state of the story and the agentic system is persisted in a database, allowing the writing process to be paused and resumed at any time.
* **Web Interface:** A simple web UI provides a real-time view of the agent's logs and the generated story content.
