import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { VERSION } from "@earendil-works/pi-coding-agent";

export default function sipzHeader(pi: ExtensionAPI) {
  pi.on("session_start", (_event, ctx) => {
    if (ctx.mode !== "tui") {
      return;
    }

    ctx.ui.setTitle("Sipz Nutrient Research Agent");
    ctx.ui.setHeader((_tui, theme) => ({
      render(width: number): string[] {
        const fit = (value: string) =>
          value.length <= width ? value : `${value.slice(0, Math.max(0, width - 3))}...`;

        return [
          theme.bold(theme.fg("accent", fit("Sipz Nutrient Research Agent"))),
          theme.fg(
            "text",
            fit("Evidence-focused research for orally consumed nutrients and bioactives"),
          ),
          theme.fg("dim", fit(`Powered by the Pi harness v${VERSION}`)),
          "",
          theme.fg(
            "muted",
            fit("Enter a substance or research question to begin  |  / commands  |  Ctrl+O resources"),
          ),
        ];
      },
      invalidate() {},
    }));
  });
}
