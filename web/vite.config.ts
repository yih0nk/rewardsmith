import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the API runs on :8000 (rewardsmith-server); vite proxies to it.
// API_TARGET points the proxy at another server instead (the pod), to try the board on real runs.
const api = process.env.API_TARGET ?? "http://localhost:8000";
const proxy = { target: api, changeOrigin: true };

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": proxy,
      "/artifacts": proxy,
    },
  },
});
