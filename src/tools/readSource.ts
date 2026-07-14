import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const workspaceRoot = resolve(process.cwd(), "workspace");

export async function readSource(relativePath: string): Promise<string> {
  const absolutePath = resolve(workspaceRoot, relativePath);

  if (!absolutePath.startsWith(workspaceRoot)) {
    throw new Error(`Refusing to read outside workspace: ${relativePath}`);
  }

  return readFile(absolutePath, "utf8");
}
