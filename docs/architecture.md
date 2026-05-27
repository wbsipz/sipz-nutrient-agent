# Architecture

The MVP uses a staged Python pipeline:

```txt
Typer CLI
  -> candidate finder
  -> claim extractor
  -> body-only claim validator
  -> quote grounding checker
  -> effect CSV row builder
  -> artifact writer
  -> audit command
```

The demo path is deterministic. Live retrieval, PDF extraction, and real LLM providers should be added behind the existing interfaces after the demo passes.

## Boundary

LLMs may propose claims, but accepted claims must be grounded in body text quotes. The validator should not use abstracts for proof.
