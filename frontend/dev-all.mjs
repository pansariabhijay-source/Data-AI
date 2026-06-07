// dev-all.mjs — launch the FastAPI backend and the Next.js dev server together.
//
// Run from the frontend dir with `npm run dev`. Starts:
//   • api  — FastAPI/uvicorn backend (api.py) on http://127.0.0.1:8000
//   • web  — Next.js dev server on http://localhost:3000 (proxies /api → backend)
//
// Zero dependencies. Ctrl+C stops both; if either exits, the other is torn down.

import { spawn, spawnSync } from "node:child_process";
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
const BACKEND_PORT = 8000;
const children = [];
let shuttingDown = false;

// Free the backend port before launching. A previous `python api.py` that didn't
// shut down cleanly (common on Windows where Ctrl+C may not reap the child) keeps
// port 8000 bound. The new backend then fails with EADDRINUSE and exits, which
// tears the whole stack down — looking like "nothing runs". Reap the squatter so
// each `npm run dev` starts from a clean slate.
function freeBackendPort(port) {
  const pids = findPortPids(port);
  if (pids.length === 0) return;
  console.log(
    `\x1b[33m[dev]\x1b[0m Port ${port} is already in use by PID(s) ${pids.join(", ")} — ` +
      `reaping stale backend before starting.`,
  );
  for (const pid of pids) killPid(pid);
}

// Returns the listening PIDs on `port` (deduped). Best-effort and cross-platform.
function findPortPids(port) {
  const pids = new Set();
  if (isWin) {
    const res = spawnSync("netstat", ["-ano", "-p", "TCP"], { encoding: "utf8" });
    if (res.status !== 0 || !res.stdout) return [];
    for (const line of res.stdout.split("\n")) {
      // e.g. "  TCP    127.0.0.1:8000   0.0.0.0:0   LISTENING   17916"
      const m = line.match(/^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$/);
      if (m && Number(m[1]) === port) pids.add(m[2]);
    }
  } else {
    const res = spawnSync("lsof", ["-ti", `tcp:${port}`, "-sTCP:LISTEN"], { encoding: "utf8" });
    if (res.status !== 0 || !res.stdout) return [];
    for (const pid of res.stdout.split("\n")) if (pid.trim()) pids.add(pid.trim());
  }
  return [...pids];
}

function killPid(pid) {
  if (isWin) spawnSync("taskkill", ["/PID", pid, "/F"], { stdio: "ignore" });
  else spawnSync("kill", ["-9", pid], { stdio: "ignore" });
}

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

// Clear any orphaned backend squatting on the port, then launch.
freeBackendPort(BACKEND_PORT);
// Backend runs from the project root so api.py's relative paths (data/, artifacts/…) resolve.
start("api", "\x1b[36m", pythonCmd, ["api.py"], ROOT);
// Frontend dev server.
start("web", "\x1b[35m", process.execPath, [nextBin, "dev"], FRONTEND_DIR);
