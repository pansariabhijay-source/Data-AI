# ── Axiom frontend: Next.js (production build) ───────────────────────────────
# Builds the app and serves it with `next start` on :3000. The /api/* proxy
# target is read from BACKEND_URL at runtime (set by docker-compose), so the
# same image works locally and in the compose network.
FROM node:20-slim

ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

WORKDIR /app

# Install deps first (cached across source changes). Dev deps are needed to build.
# The committed package.json hardcodes the Mac-only @next/swc-darwin-arm64 binary
# as a direct dependency, which fails with EBADPLATFORM on Linux. Strip that pin
# and the macOS-generated lockfile inside the image, then let npm resolve the
# correct linux SWC binary. (The repo's files are untouched, so Mac dev is
# unaffected.) Next.js pulls the right platform SWC automatically.
COPY package.json package-lock.json ./
RUN npm pkg delete "dependencies.@next/swc-darwin-arm64" \
    && rm -f package-lock.json \
    && npm install --include=dev --no-audit --no-fund

# Next.js compiles rewrite targets at BUILD time (into the route manifest), so
# BACKEND_URL must be present now, not just at runtime — otherwise the /api proxy
# bakes the default 127.0.0.1:8000 and can't reach the backend service. Defaults
# to the compose service name; override with --build-arg if you rename it.
ARG BACKEND_URL=http://backend:8000
ENV BACKEND_URL=${BACKEND_URL}

# App source, then production build.
COPY . .
RUN npm run build

EXPOSE 3000
# `next start` serves the prebuilt app; rewrites read BACKEND_URL server-side.
CMD ["npm", "run", "start"]
