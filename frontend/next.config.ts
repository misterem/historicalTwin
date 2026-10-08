import type { NextConfig } from "next";

// Fully static site: `npm run build` writes plain HTML/JS/CSS to out/, which any static host
// (Cloudflare Pages, Vercel, R2) can serve. All matching happens in the Python API.
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
