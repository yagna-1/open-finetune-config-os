import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        paper: "#f5f2ea",
        ink: "#15120d",
        ember: "#d9480f",
        teal: "#0f766e",
        marine: "#1e3a8a"
      },
      boxShadow: {
        panel: "0 24px 48px -30px rgba(21, 18, 13, 0.45)"
      }
    }
  },
  plugins: []
};

export default config;
