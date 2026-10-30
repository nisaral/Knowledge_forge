/* Apply the saved (or system) theme before first paint to avoid a flash. Kept external for the CSP. */
(function () {
  var t = null;
  try { t = localStorage.getItem("kf-theme"); } catch (e) { /* storage disabled */ }
  if (t !== "light" && t !== "dark") {
    t = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.setAttribute("data-theme", t);
})();
