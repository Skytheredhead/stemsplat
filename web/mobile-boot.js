try {
  if (localStorage.getItem('playIntro')) {
    document.documentElement.classList.add('intro');
  }
} catch (_error) {}
