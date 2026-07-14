# Sipz Nutrient Agent

An auditable literature-research agent for investigating how orally consumed nutrients,
bioactives, botanicals, additives, and ingredients affect human health.

The agent runs as an interactive terminal application on the Pi agent harness. A reasoning model
orchestrates the workflow while lower-cost worker models screen papers, extract claims, assess body
adequacy, and validate evidence in parallel.

> LLMs can propose. LLMs cannot prove. Proof requires retrievable paper text and a grounded quote.

## What It Does

Given a request such as:

```text
Research the effects of valerian root on human health when orally consumed.
Find six usable full-text papers, extract claims, validate them, and create a report.
```

the agent:

1. resolves names and scientific aliases;
2. searches structured literature indexes;
3. screens titles and abstracts for direct human oral-health relevance;
4. retrieves full text through open-access and publisher-specific fallbacks;
5. expands retrieval when usable coverage is below the requested target;
6. extracts candidate claims from each usable paper;
7. validates each claim against sanitized body text;
8. verifies supporting quotes programmatically; and
9. exports Markdown, JSON, and CSV reports with an audit trail.

Stages are independently resumable. When a user requests a later stage, the orchestrator inspects
existing artifacts, runs only missing prerequisites, and stops at the requested boundary.

## Architecture

```text
User
  |
  v
Pi orchestrator (reasoning model)
  |
  +-- inspect state / choose next stage / expand retrieval
  |
  +-- typed tools -----------------------------------------------+
       |             |              |              |             |
       v             v              v              v             v
    retrieval     screening     full text      extraction    validation
       |             |              |              |             |
       +-------------+--------------+--------------+-------------+
                                      parallel worker model calls
                                                    |
                                                    v
                                      Markdown + JSON + CSV report
```

The orchestrator owns planning and stage transitions. Deterministic Python code owns schemas,
artifact persistence, deduplication, quote matching, retry limits, stopping rules, and report
generation. Skills teach the orchestrator when and how to use each tool.

See [the architecture reference](docs/architecture.md) for the detailed component and trust model.

## Evidence Safeguards

- Only human-health evidence from oral exposure is eligible.
- Animal-only, in-vitro-only, topical, inhaled, and injected evidence is rejected.
- Different plant species are not treated as interchangeable aliases.
- Mixed interventions are rejected unless the target's contribution is separately interpretable.
- Validator context excludes title, abstract, conclusion, and references where possible.
- Every accepted claim requires a result-bearing quote found in sanitized paper body text.
- Exact quote matching alone is insufficient when the quote only describes methods or study context.
- Food-level and supplement-level exposures remain distinct.
- Failed papers and claims remain visible in structured failure artifacts.

This project supports research and evidence curation. It does not provide medical advice,
diagnosis, treatment recommendations, or personalized dosing instructions.

## Literature Sources

Candidate discovery uses structured APIs before broad web fallbacks:

- PubMed / NCBI
- Europe PMC
- OpenAlex
- Semantic Scholar
- Crossref

Full-text recovery can additionally use PMC/Europe PMC XML, Unpaywall locations, DOI and publisher
pages, configured Elsevier APIs, MDPI routes, direct PDF links, and optional local Firecrawl.
Access controls and paywalls are recorded rather than bypassed.

## Requirements

- Node.js 20 or newer
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An API key for the selected model provider

## Install

### Linux and macOS

```bash
git clone https://github.com/wbsipz/sipz-nutrient-agent.git
cd sipz-nutrient-agent
node scripts/setup.mjs
cp .env.example .env
```

Add at least one model-provider key to `.env`, then start the agent:

```bash
sipz-nutrient
```

The setup script runs `npm install`, `uv sync`, and `npm link`. To avoid global linking, use
`npm run tui` from the repository.

### Windows PowerShell

```powershell
git clone https://github.com/wbsipz/sipz-nutrient-agent.git
Set-Location sipz-nutrient-agent
node scripts/setup.mjs
Copy-Item .env.example .env
notepad .env
npm run tui
```

The setup script recognizes the Windows `py -3` launcher. If Windows blocks `npm link`, use
`npm run tui` or rerun setup from an Administrator PowerShell window.

## Model Roles

The orchestrator and worker roles can use different providers and models:

```dotenv
DEEPSEEK_API_KEY=your-key-here

ORCHESTRATOR_MODEL_PROVIDER=deepseek
ORCHESTRATOR_MODEL_ID=deepseek-v4-pro
ORCHESTRATOR_THINKING=medium

WORKER_MODEL_PROVIDER=deepseek
WORKER_MODEL_ID=deepseek-v4-flash
RESEARCH_MAX_LLM_WORKERS=5
RESEARCH_MAX_RETRIEVAL_WORKERS=10
```

Direct provider support includes DeepSeek, OpenAI, and Anthropic. OpenRouter and custom
OpenAI-compatible worker endpoints are also supported. All available retrieval credentials are
documented in [.env.example](.env.example).

Session-specific overrides are available from the terminal:

```bash
sipz-nutrient \
  --orchestrator-provider anthropic \
  --orchestrator-model claude-opus-4-8 \
  --worker-provider openai \
  --worker-model gpt-5-mini
```

## Example Requests

Start an interactive session with `sipz-nutrient`, then ask:

```text
Find 15 candidate papers about oral magnesium and human health. Screen them, but do not retrieve
full text yet.
```

```text
Retrieve full text for the retained magnesium papers. Reuse the existing screening run.
```

```text
Extract claims from the available magnesium papers, but stop before validation.
```

```text
Run a complete standard-depth workflow for valerian root with a target of 10 usable full texts.
```

The agent reports progress for each stage and prints both the absolute report directory and links
to the final files.

See [the sanitized example session](examples/sample-session.md) for an end-to-end terminal flow and
the artifact summary a user receives.

## Output

Runtime artifacts are kept outside source control:

```text
workspace/
  runs/<substance>/
    retrieval/
    screening/
    full_text/
    claims/
    validation/
  reports/<substance>/
    claims_report.md
    claims_report.json
    claims_report.csv
    report_manifest.json
```

The public table includes health effect, direction, population, oral exposure, evidence type,
validated finding, status, and source. JSON preserves grounded quotes, limitations, rejected
claims, retrieval coverage, failures, and model provenance.

## Skills And Tools

Workflow guidance lives in `agent-home/skills/`:

- `retrieval`
- `paper-screening`
- `claim-extraction`
- `claim-validation`
- `report-export`
- `web-fetch`

The Pi extension in `agent-home/extensions/literature-tools.ts` registers typed tools backed by the
Python modules under `python/src/sipz_agent/`. The adaptive `advance_research_pipeline` tool owns
prerequisite discovery, resumption, target reconciliation, and bounded expansion.

## Development

```bash
npm run typecheck
npm run test:python
npm test
npm pack --dry-run
```

`npm test` runs both TypeScript type checking and the Python pytest suite. CI runs the same command
for every push and pull request.

## Known Boundaries

- Full text may remain unavailable because of publisher authentication, paywalls, or bot controls.
- Metadata APIs can return incomplete abstracts or inconsistent identifiers.
- Model outputs are schema-validated but still require evidence grounding and appropriate human review.
- Retrieval targets are goals, not guarantees; reports disclose shortfalls and stop reasons.
- Literature coverage is not equivalent to a formal clinical guideline or exhaustive systematic review.

## License

[MIT](LICENSE)
