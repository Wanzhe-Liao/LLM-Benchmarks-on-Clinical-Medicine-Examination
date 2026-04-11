# LLM Benchmarks on Clinical Medicine Examination

This repository contains the public benchmark code and paper assets for evaluating large language models on the Clinical Medicine Comprehensive Ability examination.

## Repository scope

- Core benchmark pipeline for baseline and agentic evaluation
- Example exam file that documents the supported Markdown format
- Public manuscript assets, final figures, tables, and presentation materials

The original exam source files, OCR intermediates, local configuration with API keys, and internal working notes are intentionally excluded from version control.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a local config file from the example:

```bash
cp config.example.yaml config.yaml
```

3. Set the required environment variable, for example `OPENAI_API_KEY`.

4. Run the demo benchmark:

```bash
python main.py --config config.yaml --exam data/example_exam.md --mode baseline --limit 4
```

Outputs are written to `results/YYYYMMDD_HHMMSS/`.

## Layout

- `exam/`, `models/`, `report/`, `runner/`: benchmark pipeline
- `tools/`: figure composition and supporting analysis utilities used for the public exports
- `data/example_exam.md`: minimal example exam file
- `paper/`: manuscript, figures, tables, and presentation assets

## Configuration

`config.example.yaml` supports `${ENV_VAR}` syntax. Values written in that form are resolved from environment variables at runtime.
