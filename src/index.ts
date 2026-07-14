import "dotenv/config";

import { runResearchWorkflow } from "./orchestrator/researchWorkflow.js";

await runResearchWorkflow();
