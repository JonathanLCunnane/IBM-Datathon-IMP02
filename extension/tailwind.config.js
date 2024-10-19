/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./*.html'],
  theme: {
    extend: {
      colors: {
        background: '#D8EFD3',
        darkBackground: '#95D2B3',
        verify: '#399918',
        correct: '#A3DDCB',
        likelyCorrect: '#E8E9A1',
        likelyIncorrect: '#E6B566',
        incorrect: '#E5707E'
      }
    },
    fontFamily: {
      sans: ['Roboto', 'Arial', 'sans-serif']
    }
  },
  plugins: [],
}
