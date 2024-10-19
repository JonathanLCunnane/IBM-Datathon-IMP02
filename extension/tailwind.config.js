/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./*.html'],
  theme: {
    extend: {
      colors: {
        background: '#D8EFD3',
        darkBackground: '#95D2B3',
        verify: '#399918',
      }
    },
    fontFamily: {
      sans: ['Roboto', 'Arial', 'sans-serif']
    }
  },
  plugins: [],
}
