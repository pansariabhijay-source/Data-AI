/**
 * Lightweight client-side cookie helpers.
 *
 * The auth token + selected workspace already live in `localStorage` (via the
 * zustand-persist stores). The server-side gate in `src/proxy.ts` runs before
 * any React code and can only read **cookies**, never localStorage — so we
 * mirror just enough state into cookies for it to make routing decisions.
 * These cookies are a readable mirror of the stores, not an independent source
 * of truth.
 */

export const AUTH_COOKIE = "axiom-auth";
export const WORKSPACE_COOKIE = "axiom-workspace";

// 30 days — matches how long a returning user should stay "remembered".
const DEFAULT_MAX_AGE = 60 * 60 * 24 * 30;
const BASE_OPTS = "path=/; SameSite=Lax";

export function setCookie(name: string, value: string, maxAgeSeconds = DEFAULT_MAX_AGE): void {
  if (typeof document === "undefined") return;
  document.cookie = `${name}=${encodeURIComponent(value)}; Max-Age=${maxAgeSeconds}; ${BASE_OPTS}`;
}

export function deleteCookie(name: string): void {
  if (typeof document === "undefined") return;
  document.cookie = `${name}=; Max-Age=0; ${BASE_OPTS}`;
}
