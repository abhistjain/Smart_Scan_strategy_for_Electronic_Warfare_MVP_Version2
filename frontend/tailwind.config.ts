import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // "SIGINT ops room" palette (spec section 9.1)
        base: "#0A0E17", // near-black deep navy
        panel: "#0F1524",
        cyan: "#22D3EE", // hits / active
        amber: "#F5A623", // uncertain / warning
        slateidle: "#3B4A6B", // idle
        crimson: "#EF4444", // confirmed high-priority threat only
      },
      fontFamily: {
        mono: ["var(--font-mono)", "JetBrains Mono", "IBM Plex Mono", "monospace"],
        sans: ["var(--font-sans)", "Inter", "system-ui", "sans-serif"],
      },
      keyframes: {
        "pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 0px 0px rgba(34,211,238,0.0)" },
          "50%": { boxShadow: "0 0 16px 3px rgba(34,211,238,0.7)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        sweep: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
      },
      animation: {
        "pulse-glow": "pulse-glow 1.2s ease-in-out infinite",
        shimmer: "shimmer 2.5s linear infinite",
        sweep: "sweep 6s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
