import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the workspace root to this folder so Next doesn't mis-detect it from a
  // stray lockfile higher up the tree (e.g. C:\Users\ACER\package-lock.json).
  turbopack: {
    root: import.meta.dirname,
  },
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  // The dev server proxies /api/* to the FastAPI backend and buffers the request
  // body in memory, capped at 10MB by default. Dataset uploads are far larger
  // (e.g. creditcard.csv ~150MB), so the upload was being truncated and the
  // backend connection reset ("Backend server is unavailable"). Raise the cap.
  experimental: {
    proxyClientMaxBodySize: "300mb",
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "lh3.googleusercontent.com",
      },
    ],
  },
  async rewrites() {
    // Proxy /api/* to the FastAPI backend. Defaults to the local dev server;
    // in Docker/compose BACKEND_URL points at the backend service (e.g.
    // http://backend:8000) so the same build works in both places.
    const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
