import { runResearchWorkflow } from "../orchestrator/researchWorkflow.js";

export async function startTuiApp(): Promise<void> {
  await runResearchWorkflow();
}
