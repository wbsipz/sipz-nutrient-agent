#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import dotenv from "dotenv";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const agentDir = join(projectRoot, "agent-home");
const sessionDir = join(agentDir, "sessions");
const systemPrompt = readFileSync(join(agentDir, "SYSTEM.md"), "utf8");
const piCli = join(
  projectRoot,
  "node_modules",
  "@earendil-works",
  "pi-coding-agent",
  "dist",
  "cli.js",
);

const envPath = join(projectRoot, ".env");
const envResult = dotenv.config({
  path: envPath,
  override: true,
  quiet: true,
});

if (envResult.error) {
  console.error(`[sipz-nutrient] Could not load ${envPath}: ${envResult.error.message}`);
  console.error("Create it from .env.example and add credentials for the providers you select.");
  process.exit(1);
}

function consumeOption(args, names) {
  for (let index = 0; index < args.length; index += 1) {
    for (const name of names) {
      if (args[index] === name) {
        if (index + 1 >= args.length) {
          console.error(`[sipz-nutrient] ${name} requires a value.`);
          process.exit(2);
        }
        const value = args[index + 1];
        args.splice(index, 2);
        return value;
      }
      if (args[index].startsWith(`${name}=`)) {
        const value = args[index].slice(name.length + 1);
        args.splice(index, 1);
        return value;
      }
    }
  }
  return undefined;
}

const forwardedArgs = process.argv.slice(2);
const cliOrchestratorProvider = consumeOption(forwardedArgs, ["--orchestrator-provider", "--provider"]);
const cliOrchestratorModel = consumeOption(forwardedArgs, ["--orchestrator-model", "--model"]);
const cliOrchestratorThinking = consumeOption(forwardedArgs, ["--orchestrator-thinking", "--thinking"]);
const cliWorkerProvider = consumeOption(forwardedArgs, ["--worker-provider"]);
const cliWorkerModel = consumeOption(forwardedArgs, ["--worker-model"]);
const legacyProvider = process.env.RESEARCH_MODEL_PROVIDER?.trim();
const legacyModel = process.env.RESEARCH_MODEL_ID?.trim();

const orchestratorProvider =
  cliOrchestratorProvider?.trim() ||
  process.env.ORCHESTRATOR_MODEL_PROVIDER?.trim() ||
  legacyProvider ||
  "deepseek";
const orchestratorModel =
  cliOrchestratorModel?.trim() ||
  process.env.ORCHESTRATOR_MODEL_ID?.trim() ||
  legacyModel ||
  "deepseek-v4-pro";
const orchestratorThinking =
  cliOrchestratorThinking?.trim() || process.env.ORCHESTRATOR_THINKING?.trim() || "medium";
const workerProvider =
  cliWorkerProvider?.trim() ||
  process.env.WORKER_MODEL_PROVIDER?.trim() ||
  legacyProvider ||
  "deepseek";
const workerModel =
  cliWorkerModel?.trim() ||
  process.env.WORKER_MODEL_ID?.trim() ||
  legacyModel ||
  "deepseek-v4-flash";

const orchestratorProviders = new Set(["deepseek", "openai", "anthropic", "openrouter"]);
const workerProviders = new Set(["deepseek", "openai", "anthropic", "openai-compatible"]);
const thinkingLevels = new Set(["off", "minimal", "low", "medium", "high", "xhigh"]);
if (!orchestratorProviders.has(orchestratorProvider)) {
  console.error(`[sipz-nutrient] Unsupported orchestrator provider: ${orchestratorProvider}`);
  process.exit(2);
}
if (!workerProviders.has(workerProvider)) {
  console.error(`[sipz-nutrient] Unsupported worker provider: ${workerProvider}`);
  process.exit(2);
}
if (!thinkingLevels.has(orchestratorThinking)) {
  console.error(`[sipz-nutrient] Unsupported orchestrator thinking level: ${orchestratorThinking}`);
  process.exit(2);
}

function requiredCredential(provider, role) {
  const keyByProvider = {
    deepseek: "DEEPSEEK_API_KEY",
    openai: "OPENAI_API_KEY",
    anthropic: "ANTHROPIC_API_KEY",
    openrouter: "OPENROUTER_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
  };
  const keyName = keyByProvider[provider];
  if (
    provider === "openrouter" &&
    !process.env.OPENROUTER_API_KEY?.trim() &&
    process.env.OPENAI_API_KEY?.startsWith("sk-or-v1")
  ) {
    process.env.OPENROUTER_API_KEY = process.env.OPENAI_API_KEY;
  }
  if (keyName && !process.env[keyName]?.trim()) {
    console.error(`[sipz-nutrient] ${keyName} is required by the ${role} provider '${provider}'.`);
    process.exit(1);
  }
  if (provider === "openai-compatible" && !process.env.OPENAI_COMPATIBLE_BASE_URL?.trim()) {
    console.error("[sipz-nutrient] OPENAI_COMPATIBLE_BASE_URL is required by the worker provider 'openai-compatible'.");
    process.exit(1);
  }
}

requiredCredential(orchestratorProvider, "orchestrator");
requiredCredential(workerProvider, "worker");

process.env.ORCHESTRATOR_MODEL_PROVIDER = orchestratorProvider;
process.env.ORCHESTRATOR_MODEL_ID = orchestratorModel;
process.env.ORCHESTRATOR_THINKING = orchestratorThinking;
process.env.WORKER_MODEL_PROVIDER = workerProvider;
process.env.WORKER_MODEL_ID = workerModel;
// Older Python entrypoints continue to read these names.
process.env.RESEARCH_MODEL_PROVIDER = workerProvider;
process.env.RESEARCH_MODEL_ID = workerModel;

console.error(`[sipz-nutrient] Orchestrator: ${orchestratorProvider}/${orchestratorModel} (thinking: ${orchestratorThinking})`);
console.error(`[sipz-nutrient] Workers: ${workerProvider}/${workerModel} (max parallel: ${process.env.RESEARCH_MAX_LLM_WORKERS || "5"})`);
if (forwardedArgs.includes("--help") || forwardedArgs.includes("-h")) {
  console.error("[sipz-nutrient] Model role options:");
  console.error("  --orchestrator-provider <deepseek|openai|anthropic|openrouter>");
  console.error("  --orchestrator-model <model-id>");
  console.error("  --orchestrator-thinking <off|minimal|low|medium|high|xhigh>");
  console.error("  --worker-provider <deepseek|openai|anthropic|openai-compatible>");
  console.error("  --worker-model <model-id>");
}

const args = [
  piCli,
  "--provider",
  orchestratorProvider,
  "--model",
  orchestratorModel,
  "--thinking",
  orchestratorThinking,
  "--system-prompt",
  systemPrompt,
  "--append-system-prompt",
  join(agentDir, "AGENTS.md"),
  "--no-context-files",
  "--session-dir",
  sessionDir,
  "--extension",
  join(agentDir, "extensions", "sipz-header.ts"),
  "--extension",
  join(agentDir, "extensions", "literature-tools.ts"),
  "--skill",
  join(agentDir, "skills"),
  "--tools",
  [
    "read",
    "grep",
    "find",
    "ls",
    "retrieve_candidates",
    "screen_candidates",
    "enrich_candidate",
    "retrieve_full_text",
    "retrieve_full_text_batch",
    "inspect_retrieval_run",
    "inspect_research_state",
    "advance_research_pipeline",
    "extract_claims",
    "validate_claims",
    "export_research_report",
  ].join(","),
  ...forwardedArgs,
];

const result = spawnSync(process.execPath, args, {
  cwd: projectRoot,
  env: {
    ...process.env,
    PI_CODING_AGENT_DIR: agentDir,
  },
  stdio: "inherit",
});

if (result.error) {
  console.error(`[sipz-nutrient] Failed to start Pi: ${result.error.message}`);
  process.exitCode = 1;
} else {
  process.exitCode = result.status ?? 1;
}
