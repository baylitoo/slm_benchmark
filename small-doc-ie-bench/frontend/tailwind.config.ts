import type { Config } from "tailwindcss";

/**
 * Semantic, theme-aware color tokens backed by CSS variables (see globals.css).
 * Each token is an "R G B" triple so Tailwind's `/<alpha>` opacity syntax works
 * (e.g. `bg-accent/15`). Light/dark values flip via the `.dark` class that
 * next-themes toggles on <html>.
 */
const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Semantic tokens.
        background: "rgb(var(--background) / <alpha-value>)",
        foreground: "rgb(var(--foreground) / <alpha-value>)",
        card: "rgb(var(--card) / <alpha-value>)",
        elevated: "rgb(var(--elevated) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        input: "rgb(var(--input) / <alpha-value>)",
        ring: "rgb(var(--ring) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        "muted-foreground": "rgb(var(--muted-foreground) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-foreground": "rgb(var(--accent-foreground) / <alpha-value>)",

        // Legacy aliases kept so any un-migrated markup still resolves.
        ink: "rgb(var(--background) / <alpha-value>)",
        panel: "rgb(var(--card) / <alpha-value>)",
        edge: "rgb(var(--border) / <alpha-value>)",
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      borderRadius: {
        lg: "0.5rem", // controls, inputs, chips
        xl: "0.75rem", // 12px — cards, tiles
        "2xl": "1rem", // large containers
      },
      boxShadow: {
        // soft + subtle, never drop-heavy. The hairline border does the
        // structural work.
        xs: "0 1px 2px 0 rgb(0 0 0 / 0.04)",
        card: "0 1px 2px 0 rgb(0 0 0 / 0.03), 0 1px 1px 0 rgb(0 0 0 / 0.02)",
        elevated:
          "0 8px 24px -12px rgb(0 0 0 / 0.18), 0 2px 6px -2px rgb(0 0 0 / 0.10)",
        // restrained focus/selected ring in place of an accent "glow".
        ring: "0 0 0 1px rgb(var(--accent) / 0.35)",
      },
      letterSpacing: {
        tightish: "-0.011em", // headers
        tight: "-0.02em", // display/logo
      },
      transitionTimingFunction: {
        swift: "cubic-bezier(0.2, 0, 0, 1)", // Linear-like ease
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(2px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
        "pulse-dot": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.18s cubic-bezier(0.2,0,0,1)",
        shimmer: "shimmer 1.6s infinite",
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
