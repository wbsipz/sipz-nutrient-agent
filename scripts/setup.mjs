#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const isWindows = process.platform === "win32";

function probe(command, args) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    shell: isWindows,
  });
  if (result.error || result.status !== 0) {
    return undefined;
  }
  return `${result.stdout}${result.stderr}`.trim();
}

function check(command, args, installHint) {
  const output = probe(command, args);
  if (output === undefined) {
    console.error(`[setup] ${command} is required. ${installHint}`);
    process.exit(1);
  }
  return output;
}

function findPython() {
  const candidates = isWindows
    ? [
        { command: "py", args: ["-3", "--version"] },
        { command: "python", args: ["--version"] },
        { command: "python3", args: ["--version"] },
      ]
    : [
        { command: "python3", args: ["--version"] },
        { command: "python", args: ["--version"] },
      ];

  for (const candidate of candidates) {
    const version = probe(candidate.command, candidate.args);
    if (version !== undefined) {
      return { ...candidate, version };
    }
  }

  console.error(
    "[setup] Python 3.11 or newer is required. Install it from https://www.python.org/downloads/ and ensure it is on PATH.",
  );
  process.exit(1);
}

function run(command, args) {
  console.log(`[setup] ${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    env: process.env,
    shell: isWindows,
    stdio: "inherit",
  });
  if (result.error || result.status !== 0) {
    console.error(`[setup] ${command} failed.`);
    if (isWindows && command === "npm" && args[0] === "link") {
      console.error(
        "[setup] If Windows blocked global linking, reopen PowerShell as Administrator or use `npm run tui` from this repository.",
      );
    }
    process.exit(result.status ?? 1);
  }
}

const nodeMajor = Number.parseInt(process.versions.node.split(".")[0] ?? "0", 10);
if (nodeMajor < 20) {
  console.error(`[setup] Node.js 20 or newer is required; found ${process.version}.`);
  process.exit(1);
}

const python = findPython();
const pythonVersion = python.version;
const pythonMatch = pythonVersion.match(/Python (\d+)\.(\d+)/);
if (!pythonMatch || Number(pythonMatch[1]) < 3 || (Number(pythonMatch[1]) === 3 && Number(pythonMatch[2]) < 11)) {
  console.error(`[setup] Python 3.11 or newer is required; found ${pythonVersion}.`);
  process.exit(1);
}

check("uv", ["--version"], "Install it from https://docs.astral.sh/uv/getting-started/installation/.");
check("npm", ["--version"], "Install npm with Node.js.");

console.log(`[setup] platform: ${process.platform}`);
console.log(`[setup] node: ${process.version}`);
console.log(`[setup] python: ${python.command} (${pythonVersion})`);

run("npm", ["install"]);
run("uv", ["sync"]);
run("npm", ["link"]);

console.log("\n[setup] Complete. Create .env from .env.example, add DEEPSEEK_API_KEY, then run:");
console.log("\n  sipz-nutrient\n");
