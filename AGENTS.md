# Matmul Inspector — Codex Instructions

You are working on Matmul Inspector, a numerical kernel engineering and compiler-analysis project.

## Project Context

Before starting significant development work, read:

- `project_context.md`, if present.
- `docs/trainium1_nki_gemm_case_study.md`, when working on NKI or Trainium features.
- `README.md` for current project architecture and usage.

## Development Guidelines

- Preserve existing benchmark implementations and experimental results.
- Prefer incremental changes with tests.
- Never claim optimization improvements without measurements.
- Distinguish source-level changes, compiler IR changes, and final machine-code changes.
- Maintain CPU-only compatibility for analysis utilities.
- Do not require Trainium hardware to run ordinary unit tests.
- Do not modify CUDA or NKI kernels unless the task explicitly requires it.
- Follow existing repository structure and conventions.
- Document new commands and analysis features.

## Validation

- Run relevant tests before reporting completion.
- Clearly report tests that could not run.
- Do not commit generated NEFF, CUBIN, or profiler artifacts.
- Keep Git changes scoped to the requested task.

## Collaboration

- Explain significant architectural decisions.
- Highlight uncertainty and limitations.
- Favor readable, modular implementations.
- Keep reports reproducible and suitable for technical case studies.

## Writing Style 
Use a friendly, straightforward, conversational writing style. Prefer short sentences and fewer commas. Split long sentences instead of adding several clauses. Avoid corporate language, overly polished phrasing, and unnecessary transitions. Fix spelling and grammar without removing my natural voice. Apply this to explanations, documentation, case studies, technical reports, and drafted messages. Keep technical details precise and include the evidence and limitations needed to support claims.