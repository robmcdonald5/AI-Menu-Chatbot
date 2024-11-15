/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      raleway: ["Raleway", "sans-serif"],
    },
  },
  plugins: [require("tailwindcss-hero-patterns")],
};
