/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: '#f8f6ef', 2: '#f3eee4' },
        panel: '#ffffff',
        ink: { DEFAULT: '#1f1e1d', 2: '#5c5b58', 3: '#8a8884' },
        line: { DEFAULT: '#ecebe6', 2: '#e4e2dc' },
        clay: { DEFAULT: '#c96442', soft: '#f0ddd4' },
        ok: '#5b8a5a',
        warn: '#b7791f',
        err: '#a23a3a',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'],
        serif: ['Tiempos Headline', 'Source Serif Pro', 'Songti SC', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'ui-monospace', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
