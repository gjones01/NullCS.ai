import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT_DIR = resolve(dirname(fileURLToPath(import.meta.url)));

export default defineConfig({
  root: ROOT_DIR,
  plugins: [react()],
  server: {
    port: 5173,
  },
});
