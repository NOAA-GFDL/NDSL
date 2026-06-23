// enable logo click --> homepage
document.addEventListener("DOMContentLoaded", function () {
  const title = document.querySelector(".md-header__title");
  if (title) {
    title.style.cursor = "pointer";
    title.addEventListener("click", function () {
      document.querySelector("a.md-logo").click();
    });
  }
});
