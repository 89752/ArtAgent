import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  // 开发模式用根路径，避免与 /static 代理规则冲突；
  // 构建产物才使用 /static/dist/ 前缀，由 FastAPI 托管。
  base: command === "build" ? "/static/dist/" : "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 开发期：API 与静态资源（svg/作品图）仍由 FastAPI 提供
      "/api": "http://127.0.0.1:7860",
      "/static": "http://127.0.0.1:7860",
    },
  },
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
  },
}));
