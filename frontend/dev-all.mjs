// dev-all.mjs — launch the FastAPI backend and the Next.js dev server together.
//
// Run from the frontend dir with `npm run dev`. Starts:
//   • api  — FastAPI/uvicorn backend (api.py) on http://127.0.0.1:8000
//   • web  — Next.js dev server on http://localhost:3000 (proxies /api → backend)
//
// Zero dependencies. Ctrl+C stops both; if either exits, the other is torn down.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(FRONTEND_DIR, "..");
const isWin = process.platform === "win32";

// Prefer the project's virtualenv Python; fall back to a system Python.
const venvPython = isWin
  ? join(ROOT, "venv", "Scripts", "python.exe")
  : join(ROOT, "venv", "bin", "python");
const pythonCmd = existsSync(venvPython) ? venvPython : isWin ? "python" : "python3";

// Local Next.js CLI — invoked via the current Node so we don't rely on PATH.
const nextBin = join(FRONTEND_DIR, "node_modules", "next", "dist", "bin", "next");

const RESET = "\x1b[0m";
const children = [];
let shuttingDown = false;

function pipeLines(label, color, stream, out) {
  let buffer = "";
  stream.on("data", (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) out.write(`${color}[${label}]${RESET} ${line}\n`);
  });
}

function start(label, color, command, args, cwd) {
  const child = spawn(command, args, { cwd, stdio: ["inherit", "pipe", "pipe"], env: process.env });
  pipeLines(label, color, child.stdout, process.stdout);
  pipeLines(label, color, child.stderr, process.stderr);
  child.on("error", (err) => {
    console.error(`${color}[${label}]${RESET} failed to start: ${err.message}`);
    shutdown(1);
  });
  child.on("exit", (code) => {
    if (!shuttingDown) {
      console.log(`${color}[${label}]${RESET} exited (code ${code ?? 0}) — stopping everything.`);
      shutdown(code ?? 0);
    }
  });
  children.push(child);
  return child;
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (child.exitCode === null) {
      try {
        child.kill();
      } catch {
        /* already gone */
      }
    }
  }
  process.exit(code);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

if (!existsSync(venvPython)) {
  console.warn(
    `\x1b[33m[dev]\x1b[0m venv Python not found at ${venvPython} — falling back to "${pythonCmd}" on PATH.\n` +
      `      If the backend fails to start, create the venv and install deps (see README).\n`,
  );
}

console.log("Starting Axiom — FastAPI backend + Next.js frontend. Press Ctrl+C to stop both.\n");

// Backend runs from the project root so api.py's relative paths (data/, artifacts/…) resolve.
start("api", "\x1b[36m", pythonCmd, ["api.py"], ROOT);
// Frontend dev server.
start("web", "\x1b[35m", process.execPath, [nextBin, "dev"], FRONTEND_DIR);
