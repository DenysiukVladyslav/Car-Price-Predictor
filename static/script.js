document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const resultBox = document.getElementById("result");
  const priceSpan = document.getElementById("price");

  form.addEventListener("submit", async (e) => {
    e.preventDefault(); // перехоплюємо стандартну поведінку

    const formData = new FormData(form);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    // показуємо результат
    priceSpan.textContent = data.predicted_price;
    resultBox.style.display = "block";
  });
});
