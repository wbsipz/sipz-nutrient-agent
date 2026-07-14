import {
  createResearchSession,
  extractMessageText,
  getMessageRole,
} from "../runtime/createResearchSession.js";

export async function runResearchWorkflow(): Promise<void> {
  try {
    const { session } = await createResearchSession();
    let streamedText = false;
    let printedFinalText = false;

    const printFinalAssistantMessage = (message: any) => {
      if (printedFinalText || getMessageRole(message) !== "assistant") {
        return;
      }

      const text = extractMessageText(message);

      if (!text) {
        if (process.env.DEBUG_EVENTS === "1") {
          console.error("\n[assistant_empty_message]");
          console.error(JSON.stringify(message, null, 2));
        }
        return;
      }

      if (!streamedText) {
        process.stdout.write(text);
      }

      printedFinalText = true;
    };

    session.subscribe((event: any) => {
      if (event.type === "message_update") {
        const messageEvent = event.assistantMessageEvent;

        if (messageEvent?.type === "text_delta") {
          streamedText = true;
          process.stdout.write(messageEvent.delta);
          return;
        }

        if (process.env.DEBUG_EVENTS === "1") {
          console.error("\n[message_update]", messageEvent?.type);
          console.error(JSON.stringify(event, null, 2));
        }

        return;
      }

      if (event.type === "message_end") {
        printFinalAssistantMessage(event.message);
      }

      if (event.type === "agent_end") {
        const lastAssistantMessage = [...(event.messages ?? [])]
          .reverse()
          .find((message) => getMessageRole(message) === "assistant");

        printFinalAssistantMessage(lastAssistantMessage);
      }

      console.error("\n[event]", event.type);

      if (process.env.DEBUG_EVENTS === "1") {
        console.error(JSON.stringify(event, null, 2));
      }
    });

    console.log("[boot] sending prompt...\n");

    await session.prompt(
      "Explain your active role and list the workspace directories you can use. Be concise.",
    );

    console.log("\n\n[done]");

    session.dispose();
  } catch (err) {
    console.error("\n[error]");
    console.error(err);
    process.exitCode = 1;
  }
}
