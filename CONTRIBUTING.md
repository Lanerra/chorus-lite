# Contributing to Chorus

Thank you for considering a contribution! This project follows the guidelines in `AGENTS.md`. Below is a brief overview to get started.

## Development Setup
1. Install Python 3.12 and Docker.
2. Clone the repository and copy `.env.example` to `.env`.
3. Run `python scripts/bootstrap_chorus.py` to install dependencies, start services, and initialize the databases. Manual steps are available in `README.md`.
4. Configuration is managed through the structured config system. Environment variables are documented in `.env.example` and detailed in `docs/configuration.md`.

## Coding Standards
- Format and lint code with `ruff`.
- Type check using `mypy`.
- Include Google-style docstrings and type hints.
- Follow the import order and async guidelines described in `AGENTS.md`.

## Testing
- Use `pytest` for all tests.
- Run `ruff check src && mypy src && pytest` before opening a pull request.
- Documentation-only changes may skip tests and linters.

## Pull Requests
- Title format: `[COMPONENT] Brief description of changes` following [Conventional Commits](https://www.conventionalcommits.org/) guidelines:
  - Use `feat`, `fix`, `docs`, `chore`, `refactor`, `perf`, `test` prefixes
  - Example: `docs: add API reference documentation`
- Description requirements:
  1. Summary of changes
  2. Any prompt/logic updates with rationale
  3. Schema/configuration changes
  4. Testing performed (unit/integration)
  5. Performance considerations
  6. Link to related issues (e.g., `Closes #123`)
- Ensure all CI checks pass before requesting a review
- Include `Co-authored-by` lines for multiple contributors

For detailed information, please read `AGENTS.md`.
