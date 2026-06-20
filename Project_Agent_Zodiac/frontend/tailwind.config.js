/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        zodiac: {
          deep: '#0F172A',
          purple: '#2E1065',
          accent: '#A855F7',
          light: '#F3E8FF'
        }
      }
    },
  },
  plugins: [],
}
