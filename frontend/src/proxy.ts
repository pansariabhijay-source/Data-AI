import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Next 16 renamed the `middleware` file convention to `proxy`. Same idea: this
// runs on the server before any route renders. It is the real auth gate — it
// reads the session cookie mirrored from the auth store (see src/lib/cookies.ts).

const AUTH_COOKIE = "axiom-auth";

// /auth is the front door: an unauthenticated visitor must sign in there before
// they reach anything else — including the marketing landing page ("/"). So the
// landing page is gated too, and the desired flow is:
//   open app → /auth → (sign in) → "/" landing → /free or /enterprise.
const PROTECTED_PREFIXES = ["/", "/welcome", "/free", "/enterprise", "/pipeline", "/report"];

// The only page a logged-in user has no business sitting on — they're already in.
const PUBLIC_ONLY = ["/auth"];

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isAuthed = Boolean(request.cookies.get(AUTH_COOKIE)?.value);

  // Match a protected prefix. The "/" entry deliberately matches ONLY the exact
  // root (every other path is caught by its own, more specific prefix).
  const isProtected = PROTECTED_PREFIXES.some(
    (p) => pathname === p || (p !== "/" && pathname.startsWith(`${p}/`))
  );

  // Logged out + heading into the app → bounce to /auth, remembering the
  // intended destination so we can return them there after they sign in.
  if (isProtected && !isAuthed) {
    const url = new URL("/auth", request.url);
    if (pathname !== "/") url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  // NOTE: We intentionally do NOT redirect authenticated users away from /auth
  // here. The cookie may hold a stale/expired token that the backend no longer
  // recognizes. The client-side AppShell validates the token and handles this
  // routing after confirming the token is actually valid.

  return NextResponse.next();
}

export const config = {
  // Run on everything except API routes, Next internals, and static assets
  // (anything with a file extension). `_next/data` is still covered by design.
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt|.*\\..*).*)",
  ],
};
