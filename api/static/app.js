const form = document.getElementById("pred-form");
const result = document.getElementById("result");
const riskEl = document.getElementById("risk");
const probEl = document.getElementById("prob");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    time_in_hospital: Number(document.getElementById("time_in_hospital").value),
    n_medications:    Number(document.getElementById("n_medications").value),
    n_procedures:     Number(document.getElementById("n_procedures").value),
    diag_group:       document.getElementById("diag_group").value,
  };

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Prediction failed");

    const pct = (data.prob * 100).toFixed(1) + "%";
    riskEl.textContent = `Risk: ${data.risk.toUpperCase()}`;
    riskEl.className = "risk " + (data.risk === "high" ? "high" : "low");
    probEl.textContent = `Probability: ${pct}`;
    result.classList.remove("hidden");
  } catch (err) {
    riskEl.textContent = "Error: " + err.message;
    riskEl.className = "risk high";
    probEl.textContent = "";
    result.classList.remove("hidden");
  }
});
