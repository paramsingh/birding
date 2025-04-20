# KODING.md for birdclef-2025

This file provides context for AI coding agents working in this repository.

## Project Overview

This project is for the BirdCLEF 2025 Kaggle competition, focusing on identifying species from audio soundscapes using Python.

## Build/Test Commands

*   **Activate venv:** `source bin/activate`
*   **Install dependencies:** `pip install -r requirements.txt` (assuming a requirements file exists or will be created)
*   **Run Python script:** `python path/to/script.py`
*   **Linting:** (No specific linter found yet - consider adding `ruff` or `flake8`)
*   **Testing:** (No specific test framework found yet - consider adding `pytest`)
    *   Run all tests: `pytest`
    *   Run specific test file: `pytest path/to/test_file.py`
    *   Run specific test: `pytest path/to/test_file.py::test_name`

## Code Style Guidelines

*   **Formatting:** Follow PEP 8 guidelines. Use an auto-formatter like `black` or `ruff format`.
*   **Imports:** Group imports (standard library, third-party, local modules). Use `isort` or `ruff --select I` for sorting.
*   **Typing:** Use type hints (PEP 484) for function signatures and variables.
*   **Naming:** Use `snake_case` for variables and functions, `PascalCase` for classes.
*   **Error Handling:** Use standard Python exceptions. Provide informative error messages.
*   **Docstrings:** Use Google-style docstrings for modules, classes, and functions.
*   **Notebooks:** If using Jupyter notebooks, keep cells concise and add markdown for explanations. Clear outputs before committing.
*   **Dependencies:** Add new dependencies to `requirements.txt`.

## Development Workflow

For any significant work:

1.  **Write Tests:** Implement tests *before* writing or changing the corresponding code (Test-Driven Development approach).
2.  **Implement Code:** Write or modify the code to meet the requirements and pass the tests.
3.  **Create Notebook (Optional):** If relevant, create a Jupyter notebook to demonstrate the functionality, ensuring clear explanations and aesthetics. Clear outputs before committing.

## Codebase Structure

*   `data/`: Contains competition data (ignored by git).
*   `notes/`: Research and exploratory notes.
*   `src/`: (Recommended) Place source code here (e.g., data processing, model training, inference scripts).
*   `notebooks/`: (Recommended) Place Jupyter notebooks here.
*   `bin/`, `lib/`, `include/`: Virtual environment files (ignored by git).
