import { resolve } from "node:path";
import { spawn } from "node:child_process";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type ToolEnvelope = {
  ok: boolean;
  tool?: string;
  result?: unknown;
  error?: { type?: string; message?: string };
};

type ToolUpdate = (update: {
  content: Array<{ type: "text"; text: string }>;
  details: Record<string, unknown>;
}) => void;

type WorkingUi = {
  setWorkingMessage(message?: string): void;
};

async function runBridge(
  cwd: string,
  toolName: string,
  payload: unknown,
  signal?: AbortSignal,
  onUpdate?: ToolUpdate,
  ui?: WorkingUi,
): Promise<ToolEnvelope> {
  const projectRoot = resolve(cwd);
  return new Promise((resolvePromise, reject) => {
    const child = spawn(
      "uv",
      ["run", "python", "-m", "sipz_agent.tools.literature_bridge", toolName],
      {
        cwd: projectRoot,
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, SIPZ_PROGRESS_JSONL: "1" },
      },
    );
    let stdout = "";
    let stderr = "";
    let stderrBuffer = "";
    const startedAt = Date.now();
    const updateProgress = (message: string, details: Record<string, unknown>) => {
      ui?.setWorkingMessage(message);
      onUpdate?.({ content: [{ type: "text", text: message }], details });
    };
    updateProgress(`Starting ${toolName.replaceAll("_", " ")}...`, { stage: toolName });
    const heartbeat = setInterval(() => {
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
      updateProgress(`${toolName.replaceAll("_", " ")} still running (${elapsedSeconds}s)...`, {
        stage: toolName,
        elapsed_seconds: elapsedSeconds,
        heartbeat: true,
      });
    }, 15_000);
    const abort = () => child.kill("SIGTERM");
    signal?.addEventListener("abort", abort, { once: true });
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      stderrBuffer += chunk;
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line) as {
            type?: string;
            message?: string;
            details?: Record<string, unknown>;
          };
          if (event.type === "progress" && event.message) {
            updateProgress(event.message, event.details ?? {});
          }
        } catch {
          // Preserve non-progress stderr for error diagnostics.
        }
      }
    });
    child.on("error", (error) => {
      ui?.setWorkingMessage();
      reject(error);
    });
    child.on("close", (code) => {
      clearInterval(heartbeat);
      ui?.setWorkingMessage();
      signal?.removeEventListener("abort", abort);
      try {
        const response = JSON.parse(stdout.trim()) as ToolEnvelope;
        if (!response.ok) {
          reject(new Error(response.error?.message ?? `Tool ${toolName} failed`));
          return;
        }
        resolvePromise(response);
      } catch (error) {
        reject(
          new Error(
            `Tool ${toolName} returned invalid JSON (exit ${code}): ${stderr || stdout}`,
            { cause: error },
          ),
        );
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

function resultContent(response: ToolEnvelope) {
  const result = response.result as Record<string, unknown> | undefined;
  if (typeof result?.terminal_response_markdown === "string") {
    return {
      content: [{ type: "text" as const, text: result.terminal_response_markdown }],
      details: response.result ?? {},
    };
  }
  let contentResult: unknown = response.result;
  if (result && Array.isArray(result.candidates) && result.output_path) {
    const { candidates: _candidates, ...summary } = result;
    contentResult = {
      ...summary,
      candidates_path: result.output_path,
      note: "Candidate records were written to candidates_path; pass that path to screen_candidates.",
    };
  }
  return {
    content: [{ type: "text" as const, text: JSON.stringify(contentResult, null, 2) }],
    details: response.result ?? {},
  };
}

const candidateFields = {
  id: Type.Optional(Type.String()),
  title: Type.String({ minLength: 1 }),
  doi: Type.Optional(Type.String()),
  pmid: Type.Optional(Type.String()),
  url: Type.Optional(Type.String()),
  source: Type.Optional(Type.String()),
  retrieval_query: Type.Optional(Type.String()),
  abstract: Type.Optional(Type.String()),
  year: Type.Optional(Type.Integer()),
  selection_reason: Type.Optional(Type.String()),
  page_summary: Type.Optional(Type.String()),
  body_text: Type.Optional(Type.String()),
};

const candidateSchema = Type.Object(candidateFields);

export default function literatureTools(pi: ExtensionAPI) {
  pi.registerTool({
    name: "retrieve_candidates",
    label: "Retrieve Literature Candidates",
    description:
      "Searches prioritized scholarly sources with pagination, metadata enrichment, deduplication, provenance, and partial-failure reporting.",
    promptSnippet: "Retrieve candidate papers for oral-ingestion human-health research",
    promptGuidelines: [
      "Use retrieve_candidates for literature discovery; do not treat returned candidates as validated health evidence.",
    ],
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      aliases: Type.Optional(Type.Array(Type.String())),
      queries: Type.Optional(Type.Array(Type.String())),
      depth: Type.Optional(
        Type.Union([Type.Literal("light"), Type.Literal("standard"), Type.Literal("deep")]),
      ),
      target_count: Type.Optional(Type.Integer({ minimum: 1, maximum: 200 })),
      max_pages: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
      page_size: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
      output_path: Type.Optional(Type.String()),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Searching literature for ${params.substance}...` }],
        details: { stage: "retrieval" },
      });
      return resultContent(await runBridge(ctx.cwd, "retrieve_candidates", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "screen_candidates",
    label: "Screen Literature Candidates",
    description:
      "Classifies every candidate from title and abstract, retains only direct human studies and focused reviews, rejects indirect or confounded records, reconciles counts, and optionally writes screening artifacts.",
    promptSnippet: "Screen a retrieved candidate set before attempting full-text retrieval",
    promptGuidelines: [
      "Always use screen_candidates after retrieve_candidates when screening is requested; do not replace it with an improvised Markdown table or silently omit candidates.",
      "Only pass screen_candidates retained results to retrieve_full_text unless the user explicitly requests background sources.",
    ],
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      aliases: Type.Optional(Type.Array(Type.String())),
      candidates: Type.Optional(Type.Array(candidateSchema, { minItems: 1 })),
      candidates_path: Type.Optional(Type.String()),
      output_dir: Type.Optional(Type.String()),
      max_candidates: Type.Optional(Type.Integer({ minimum: 1, maximum: 200 })),
      resume: Type.Optional(Type.Boolean()),
      max_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [
          {
            type: "text",
            text: params.candidates
              ? `Screening ${params.candidates.length} candidate papers...`
              : `Screening candidates from ${params.candidates_path}...`,
          },
        ],
        details: { stage: "paper_screening", total: params.candidates?.length },
      });
      return resultContent(await runBridge(ctx.cwd, "screen_candidates", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "enrich_candidate",
    label: "Enrich Literature Candidate",
    description:
      "Recovers missing abstract and identifier metadata for a candidate using DOI, PMID, and scholarly metadata services.",
    promptSnippet: "Enrich a paper candidate that lacks abstract or identifier metadata",
    parameters: Type.Object(candidateFields),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Enriching metadata for ${params.title}...` }],
        details: { stage: "metadata_enrichment" },
      });
      return resultContent(await runBridge(ctx.cwd, "enrich_candidate", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "retrieve_full_text",
    label: "Retrieve Paper Full Text",
    description:
      "Attempts publisher, open-access, Europe PMC, identifier, API, and configured scraping routes and reports every full-text retrieval attempt.",
    promptSnippet: "Retrieve and optionally save the full text of a known candidate paper",
    promptGuidelines: [
      "Use retrieve_full_text only after candidate discovery; a successful download does not validate a health claim.",
    ],
    parameters: Type.Object({
      ...candidateFields,
      body_text: Type.Optional(Type.String()),
      output_dir: Type.Optional(Type.String()),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Retrieving full text for ${params.title}...` }],
        details: { stage: "full_text_retrieval" },
      });
      return resultContent(await runBridge(ctx.cwd, "retrieve_full_text", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "retrieve_full_text_batch",
    label: "Retrieve Retained Paper Full Texts",
    description:
      "Retrieves full text for a retained-source artifact, resumes prior work, skips successful existing texts, and writes a manifest plus attempt log.",
    promptSnippet: "Retrieve full texts for every retained paper in a screening artifact",
    promptGuidelines: [
      "Prefer this batch tool over repeated retrieve_full_text calls when a retained_sources.json artifact exists.",
    ],
    parameters: Type.Object({
      retained_sources_path: Type.String({ minLength: 1 }),
      output_dir: Type.String({ minLength: 1 }),
      resume: Type.Optional(Type.Boolean()),
      retry_failed: Type.Optional(Type.Boolean()),
      max_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      max_attempts: Type.Optional(Type.Integer({ minimum: 1, maximum: 3 })),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: "Retrieving retained paper full texts..." }],
        details: { stage: "full_text_retrieval" },
      });
      return resultContent(await runBridge(ctx.cwd, "retrieve_full_text_batch", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "inspect_retrieval_run",
    label: "Inspect Retrieval Run",
    description:
      "Reads an existing research run and summarizes queries, pagination, source counts, failures, stop reason, and full-text statuses.",
    promptSnippet: "Diagnose an existing literature retrieval run",
    parameters: Type.Object({ run_path: Type.String({ minLength: 1 }) }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Inspecting ${params.run_path}...` }],
        details: { stage: "run_inspection" },
      });
      return resultContent(await runBridge(ctx.cwd, "inspect_retrieval_run", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "inspect_research_state",
    label: "Inspect Subject Research State",
    description:
      "Finds existing retrieval, screening, and full-text artifacts for a subject and recommends the next prerequisite-aware action.",
    promptSnippet: "Locate reusable research artifacts before executing a requested stage",
    promptGuidelines: [
      "Call this first when the user asks to continue or run a specific stage without providing a run path.",
    ],
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      workspace_root: Type.Optional(Type.String()),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Inspecting existing ${params.substance} research...` }],
        details: { stage: "state_inspection" },
      });
      return resultContent(await runBridge(ctx.cwd, "inspect_research_state", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "advance_research_pipeline",
    label: "Advance Research to Requested Stage",
    description:
      "Runs one resumable retrieval/coverage round and advances only when the screened-paper or usable-full-text target is met or deterministic search limits are exhausted.",
    promptSnippet: "Advance subject research only through the user's requested stage",
    promptGuidelines: [
      "Use this for subject-level paper-list or full-text requests when exact artifact paths were not supplied.",
      "When expansion_recommended is true, inspect the coverage fields, propose 2-4 new non-duplicate human oral-health queries, and call this tool again with expansion_queries.",
      "Do not proceed to extraction or validation while expansion_recommended is true.",
      "Omit workspace_root to use the standard workspace directory; do not pass the repository root.",
    ],
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      target_stage: Type.Union([
        Type.Literal("screening"),
        Type.Literal("full_text"),
        Type.Literal("claim_extraction"),
        Type.Literal("claim_validation"),
      ]),
      aliases: Type.Optional(Type.Array(Type.String())),
      expansion_queries: Type.Optional(Type.Array(Type.String())),
      requested_count: Type.Optional(Type.Integer({ minimum: 1, maximum: 200 })),
      depth: Type.Optional(
        Type.Union([Type.Literal("light"), Type.Literal("standard"), Type.Literal("deep")]),
      ),
      workspace_root: Type.Optional(Type.String({ description: "Workspace container; omit for the standard workspace directory." })),
      resume: Type.Optional(Type.Boolean()),
      screening_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      full_text_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      extraction_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      validation_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      retry_failed_validation: Type.Optional(Type.Boolean()),
      max_expansion_rounds: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      max_candidates: Type.Optional(Type.Integer({ minimum: 10, maximum: 200 })),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Advancing ${params.substance} through ${params.target_stage}...` }],
        details: { stage: params.target_stage },
      });
      return resultContent(await runBridge(ctx.cwd, "advance_research_pipeline", params, signal, onUpdate, ctx.ui));
    },
  });

  const artifactPath = Type.String({ minLength: 1 });
  pi.registerTool({
    name: "extract_claims",
    label: "Extract Paper Claims",
    description:
      "Extracts structured oral human-health claim proposals from retrieved paper body text and writes auditable artifacts.",
    promptSnippet: "Extract candidate claims from retained papers with usable full text",
    promptGuidelines: [
      "Use this tool instead of manually drafting claims from abstracts or conversational summaries.",
    ],
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      sources_path: artifactPath,
      raw_texts_manifest_path: artifactPath,
      raw_texts_dir: artifactPath,
      output_dir: artifactPath,
      resume: Type.Optional(Type.Boolean()),
      max_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Extracting claims for ${params.substance}...` }],
        details: { stage: "claim_extraction" },
      });
      return resultContent(await runBridge(ctx.cwd, "extract_claims", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "validate_claims",
    label: "Validate Paper Claims",
    description:
      "Assesses paper-body adequacy, holds insufficient texts for retrieval recovery, validates eligible proposed claims, and deterministically grounds supporting quotes.",
    promptSnippet: "Validate proposed claims using paper body text, excluding abstract and references",
    promptGuidelines: [
      "Never substitute manual validation for this tool; accepted claims require grounded body-text quotes.",
      "When artifact paths are known, call this tool directly and use its reconciled counts instead of pre-reading large artifacts or inferring counts from truncated output.",
    ],
    parameters: Type.Object({
      proposed_claims_path: artifactPath,
      sources_path: artifactPath,
      raw_texts_manifest_path: artifactPath,
      raw_texts_dir: artifactPath,
      output_dir: artifactPath,
      max_body_chars: Type.Optional(Type.Integer({ minimum: 1 })),
      resume: Type.Optional(Type.Boolean()),
      retry_failed: Type.Optional(Type.Boolean()),
      max_workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: "Validating proposed claims against body text..." }],
        details: { stage: "claim_validation" },
      });
      return resultContent(await runBridge(ctx.cwd, "validate_claims", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "export_research_report",
    label: "Export Research Report",
    description:
      "Writes a deterministic public claims report in Markdown, JSON, and CSV and returns a terminal-ready table plus report link.",
    promptSnippet: "Export a completed validated research run",
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      proposed_claims_path: artifactPath,
      validated_claims_path: artifactPath,
      rejected_claims_path: artifactPath,
      sources_path: artifactPath,
      output_dir: artifactPath,
      held_claims_path: Type.Optional(artifactPath),
      body_adequacy_path: Type.Optional(artifactPath),
      validation_failures_path: Type.Optional(artifactPath),
      retrieval_expansion_path: Type.Optional(artifactPath),
    }),
    promptGuidelines: [
      "Print terminal_response_markdown verbatim as the entire response; do not preface, recalculate, shorten, or rewrite it.",
    ],
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Exporting ${params.substance} research report...` }],
        details: { stage: "report_export" },
      });
      return resultContent(await runBridge(ctx.cwd, "export_research_report", params, signal, onUpdate, ctx.ui));
    },
  });

  pi.registerTool({
    name: "run_research_pipeline",
    label: "Run Nutrient Research Pipeline",
    description:
      "Legacy single-call compatibility workflow. Pi should use the adaptive advance_research_pipeline loop and export_research_report instead.",
    promptSnippet: "Run the complete oral human-health research pipeline for a substance",
    promptGuidelines: [
      "Do not use this tool from Pi; use advance_research_pipeline through claim_validation and then export_research_report.",
      "Never use this tool for retrieval-only, screening-only, full-text-only, extraction-only, or validation-only requests.",
      "On completion, print terminal_response_markdown verbatim as the entire response; do not preface, recalculate, shorten, or rewrite it.",
      "Report completion only from returned artifact paths and reconciled counts.",
    ],
    parameters: Type.Object({
      substance: Type.String({ minLength: 1 }),
      aliases: Type.Optional(Type.Array(Type.String())),
      depth: Type.Optional(
        Type.Union([Type.Literal("light"), Type.Literal("standard"), Type.Literal("deep")]),
      ),
      output_root: Type.Optional(Type.String()),
      run_id: Type.Optional(Type.String()),
      target_count: Type.Optional(Type.Integer({ minimum: 1, maximum: 200 })),
      max_pages: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
      page_size: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
      resume: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      onUpdate?.({
        content: [{ type: "text", text: `Running complete research pipeline for ${params.substance}...` }],
        details: { stage: "research_pipeline" },
      });
      return resultContent(await runBridge(ctx.cwd, "run_research_pipeline", params, signal, onUpdate, ctx.ui));
    },
  });
}
