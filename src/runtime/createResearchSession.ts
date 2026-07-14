import { readFile } from "node:fs/promises";

import {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
} from "@earendil-works/pi-coding-agent";

const defaultSystemPrompt = `
You are a research workflow agent.

You are not a coding assistant.
Your job is to search sources, read documents, extract claims, compare evidence,
register citations, and produce cited research notes or reports.

Do not review, refactor, or modify code unless explicitly asked.
`.trim();

const defaultAgentsMd = `
# Research Agent Context

- workspace/sources: original source files
- workspace/extracted: extracted source text
- workspace/notes: scratch notes
- workspace/reports: final outputs

Rules:
- Search before answering.
- Read sources before using them.
- Never cite a source you have not read.
- Do not default to coding-agent behavior.
`.trim();

export type ResearchSessionRuntime = {
  session: Awaited<ReturnType<typeof createAgentSession>>["session"];
  cwd: string;
  agentDir: string;
  sessionDir: string;
  selectedModel: {
    provider: string;
    id: string;
  };
};

async function readText(path: string, fallback: string): Promise<string> {
  try {
    return await readFile(path, "utf8");
  } catch {
    return fallback;
  }
}

export function extractMessageText(message: any): string {
  if (typeof message?.errorMessage === "string") {
    return message.errorMessage;
  }

  const content = message?.content;

  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }

      if (part?.type === "text" && typeof part.text === "string") {
        return part.text;
      }

      if (typeof part?.text === "string") {
        return part.text;
      }

      return "";
    })
    .join("");
}

export function getMessageRole(message: any): string | undefined {
  return message?.role;
}

export async function createResearchSession(): Promise<ResearchSessionRuntime> {
  const cwd = process.cwd();
  const agentDir = `${cwd}/agent-home`;
  const sessionDir = `${agentDir}/sessions`;

  console.log("[boot] cwd:", cwd);
  console.log("[boot] agentDir:", agentDir);
  console.log("[boot] sessionDir:", sessionDir);
  console.log("[boot] setting up auth/model registry...");

  const authStorage = AuthStorage.inMemory();
  const modelRegistry = ModelRegistry.create(authStorage, `${agentDir}/models.json`);
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY;
  const openAiApiKey = process.env.OPENAI_API_KEY;
  const openRouterApiKey =
    process.env.OPENROUTER_API_KEY ??
    (openAiApiKey?.startsWith("sk-or-v1") ? openAiApiKey : undefined);

  if (deepSeekApiKey) {
    console.log("[boot] DEEPSEEK_API_KEY detected");
    authStorage.setRuntimeApiKey("deepseek", deepSeekApiKey);
  }

  if (openAiApiKey) {
    console.log(
      "[boot] OPENAI_API_KEY detected",
      openAiApiKey.startsWith("sk-or-v1") ? "(looks like OpenRouter)" : "",
    );
    if (!openAiApiKey.startsWith("sk-or-v1")) {
      authStorage.setRuntimeApiKey("openai", openAiApiKey);
    }
  }

  if (process.env.ANTHROPIC_API_KEY) {
    console.log("[boot] ANTHROPIC_API_KEY detected");
    authStorage.setRuntimeApiKey("anthropic", process.env.ANTHROPIC_API_KEY);
  }

  if (openRouterApiKey) {
    console.log("[boot] OpenRouter key detected");
    authStorage.setRuntimeApiKey("openrouter", openRouterApiKey);
  }

  const availableModels = await modelRegistry.getAvailable();

  console.log(
    `[boot] available models: ${availableModels.length}`,
    availableModels
      .slice(0, 20)
      .map((model: any) => `${model.provider}/${model.id ?? model.modelId ?? model.name}`)
      .join(", ") + (availableModels.length > 20 ? ", ..." : "") || "(none)",
  );

  if (availableModels.length === 0) {
    throw new Error(
      "No available model found. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in your shell/.env, or configure Pi auth.",
    );
  }

  const preferredProvider =
    process.env.ORCHESTRATOR_MODEL_PROVIDER ??
    process.env.RESEARCH_MODEL_PROVIDER ??
    (deepSeekApiKey ? "deepseek" : openRouterApiKey ? "openrouter" : "openai");
  const preferredModelId =
    process.env.ORCHESTRATOR_MODEL_ID ??
    process.env.RESEARCH_MODEL_ID ??
    (preferredProvider === "deepseek"
      ? "deepseek-v4-pro"
      : preferredProvider === "openrouter"
        ? "openai/gpt-4o-mini"
        : "gpt-4o-mini");

  const selectedModel =
    modelRegistry.find(preferredProvider, preferredModelId) ??
    (deepSeekApiKey && preferredProvider !== "deepseek"
      ? modelRegistry.find("deepseek", "deepseek-v4-pro")
      : undefined) ??
    (openRouterApiKey && preferredProvider !== "openrouter"
      ? modelRegistry.find("openrouter", "openai/gpt-4o-mini")
      : undefined) ??
    (!deepSeekApiKey && !openRouterApiKey && preferredProvider !== "openai"
      ? modelRegistry.find("openai", "gpt-4o-mini")
      : undefined) ??
    availableModels.find(
      (model: any) =>
        model.provider === preferredProvider && model.id === preferredModelId,
    ) ??
    availableModels[0];

  if (!selectedModel) {
    throw new Error("No selected model available.");
  }

  console.log(
    "[boot] selected model:",
    `${selectedModel.provider}/${selectedModel.id}`,
  );
  console.log(
    "[boot] worker model:",
    `${process.env.WORKER_MODEL_PROVIDER ?? process.env.RESEARCH_MODEL_PROVIDER ?? "deepseek"}/${process.env.WORKER_MODEL_ID ?? process.env.RESEARCH_MODEL_ID ?? "deepseek-v4-flash"}`,
  );
  process.env.ORCHESTRATOR_MODEL_PROVIDER = selectedModel.provider;
  process.env.ORCHESTRATOR_MODEL_ID = selectedModel.id;
  process.env.ORCHESTRATOR_THINKING ??= "medium";
  process.env.WORKER_MODEL_PROVIDER ??= process.env.RESEARCH_MODEL_PROVIDER ?? "deepseek";
  process.env.WORKER_MODEL_ID ??= process.env.RESEARCH_MODEL_ID ?? "deepseek-v4-flash";
  process.env.RESEARCH_MODEL_PROVIDER = process.env.WORKER_MODEL_PROVIDER;
  process.env.RESEARCH_MODEL_ID = process.env.WORKER_MODEL_ID;

  const systemPrompt = await readText(`${agentDir}/SYSTEM.md`, defaultSystemPrompt);
  const agentsMd = await readText(`${agentDir}/AGENTS.md`, defaultAgentsMd);

  console.log("[boot] loading custom resource loader...");

  const loader = new DefaultResourceLoader({
    cwd,
    agentDir,
    additionalExtensionPaths: [
      `${agentDir}/extensions/sipz-header.ts`,
      `${agentDir}/extensions/literature-tools.ts`,
    ],
    noPromptTemplates: true,
    noContextFiles: true,
    systemPromptOverride: () => systemPrompt,
    appendSystemPromptOverride: () => [],
    agentsFilesOverride: () => ({
      agentsFiles: [
        {
          path: "/virtual/AGENTS.md",
          content: agentsMd,
        },
      ],
    }),
    promptsOverride: () => ({
      prompts: [],
      diagnostics: [],
    }),
  });

  await loader.reload();

  console.log("[boot] creating session...");

  const { session } = await createAgentSession({
    cwd,
    agentDir,
    resourceLoader: loader,
    sessionManager: SessionManager.create(cwd, sessionDir),
    authStorage,
    modelRegistry,
    model: selectedModel,
    thinkingLevel: (process.env.ORCHESTRATOR_THINKING ?? "medium") as any,
    tools: [
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
    ],
  });

  return {
    session,
    cwd,
    agentDir,
    sessionDir,
    selectedModel: {
      provider: selectedModel.provider,
      id: selectedModel.id,
    },
  };
}
