import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        // SSE 必须关闭代理超时
        proxyTimeout: 0,
        timeout: 0,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
