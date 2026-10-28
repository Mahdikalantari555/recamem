import type { Config } from "tailwindcss";

// Design tokens from docs/ui-design.md §4. Components reference the token names,
// never raw hex values, so the identity can change in one place.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#111318",
        surface: "#1D2027",
        "surface-raised": "#252933",
        border: "#343943",
        "text-primary": "#F3F4F6",
        "text-secondary": "#9CA3AF",
        accent: "#B5F36D",
        success: "#8ED081",
        warning: "#F0C674",
        danger: "#F07878",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: {
        card: "12px",
        control: "8px",
      },
      spacing: {
        1: "4px",
        2: "8px",
        3: "12px",
        4: "16px",
        5: "24px",
        6: "32px",
        7: "48px",
      },
    },
  },
  plugins: [],
} satisfies Config;
