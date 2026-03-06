/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        panel: "#0f1113",
        neon: "#b7f5ff",
        accent: "#d5e8ef",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(226,232,240,0.28), 0 0 36px rgba(186,230,253,0.12)",
      },
    },
  },
  plugins: [],
};
