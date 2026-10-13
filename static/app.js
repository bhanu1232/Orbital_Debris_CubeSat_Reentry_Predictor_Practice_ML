// AstroDecay ML - Frontend Controller

let missionPresets = {};

// Elements
const form = document.getElementById("orbitForm");
const submitBtn = document.getElementById("submitBtn");
const backendStatus = document.getElementById("backendStatus");
const badgeStatus = document.getElementById("badgeStatus");

const predDaysEl = document.getElementById("predDays");
const predYearsEl = document.getElementById("predYears");
const riskClassEl = document.getElementById("riskClass");
const complianceStatusEl = document.getElementById("complianceStatus");
const decayRateEl = document.getElementById("decayRate");
const meanAltEl = document.getElementById("meanAlt");
const eccentricityEl = document.getElementById("eccentricity");
const solarRegimeEl = document.getElementById("solarRegime");
const guidanceTextEl = document.getElementById("guidanceText");
const guidanceBox = document.getElementById("guidanceBox");

// Inputs
const objectTypeInput = document.getElementById("objectType");
const perigeeInput = document.getElementById("perigee");
const apogeeInput = document.getElementById("apogee");
const inclinationInput = document.getElementById("inclination");
const ballisticCoeffInput = document.getElementById("ballisticCoeff");
const solarFluxInput = document.getElementById("solarFlux");
const rcsInput = document.getElementById("rcs");

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
  await checkHealth();
  await loadPresets();
  // Run initial prediction with default values
  triggerPrediction();
});

// Check API health and load metrics
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      backendStatus.textContent = "FastAPI ML Core Active";
      if (data.metrics) {
        document.getElementById("r2Score").textContent = data.metrics.r2_score;
        document.getElementById("modelAccuracy").textContent = data.metrics.accuracy + "%";
      }
    } else {
      backendStatus.textContent = "API Error";
    }
  } catch (err) {
    backendStatus.textContent = "Offline / Connecting...";
  }
}

// Load presets from backend
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    if (res.ok) {
      missionPresets = await res.json();
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

// Apply selected preset
function applyPreset(key) {
  const p = missionPresets[key];
  if (!p) return;

  objectTypeInput.value = p.object_type;
  perigeeInput.value = p.perigee_km;
  apogeeInput.value = p.apogee_km;
  inclinationInput.value = p.inclination_deg;
  ballisticCoeffInput.value = p.ballistic_coeff;
  solarFluxInput.value = p.solar_flux_f107;
  rcsInput.value = p.rcs_m2;

  // Flash preset button feedback
  triggerPrediction();
}

// Validate that apogee is not smaller than perigee
perigeeInput.addEventListener("change", () => {
  if (parseFloat(apogeeInput.value) < parseFloat(perigeeInput.value)) {
    apogeeInput.value = perigeeInput.value;
  }
});

// Form submission handler
form.addEventListener("submit", (e) => {
  e.preventDefault();
  triggerPrediction();
});

async function triggerPrediction() {
  const perigee = parseFloat(perigeeInput.value);
  const apogee = parseFloat(apogeeInput.value);

  if (apogee < perigee) {
    alert("Apogee altitude must be greater than or equal to perigee altitude.");
    return;
  }

  const payload = {
    object_type: objectTypeInput.value,
    perigee_km: perigee,
    apogee_km: apogee,
    inclination_deg: parseFloat(inclinationInput.value),
    ballistic_coeff: parseFloat(ballisticCoeffInput.value),
    solar_flux_f107: parseFloat(solarFluxInput.value),
    rcs_m2: parseFloat(rcsInput.value)
  };

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span>⏳ Computing Trajectory...</span>`;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || "Prediction request failed.");
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<span class="btn-icon">⚡</span><span>Predict Orbital Reentry</span>`;
  }
}

function renderResults(data) {
  const pred = data.prediction;
  const tel = data.telemetry_summary;

  // Animate primary counter
  animateNumber(predDaysEl, pred.lifespan_days);
  predYearsEl.textContent = `Equiv. ~${pred.lifespan_years} Years in Orbit`;

  // Status Badge
  badgeStatus.textContent = pred.status_badge;
  badgeStatus.style.color = pred.status_color;
  badgeStatus.style.border = `1px solid ${pred.status_color}`;
  badgeStatus.style.background = `${pred.status_color}18`;

  // Risk Class
  riskClassEl.textContent = pred.risk_category;
  riskClassEl.style.color = pred.status_color;

  // Compliance
  if (pred.fcc_5yr_compliant) {
    complianceStatusEl.textContent = "COMPLIANT (≤ 5 Years)";
    complianceStatusEl.style.color = "var(--accent-green)";
  } else {
    complianceStatusEl.textContent = "NON-COMPLIANT (> 5 Years)";
    complianceStatusEl.style.color = "var(--accent-rose)";
  }

  // Secondary telemetry
  decayRateEl.textContent = `${pred.daily_decay_rate_meters} m/day`;
  meanAltEl.textContent = `${tel.mean_altitude_km} km`;
  eccentricityEl.textContent = tel.eccentricity;
  solarRegimeEl.textContent = tel.solar_activity_regime;

  // Guidance box
  guidanceTextEl.textContent = pred.action_recommendation;
}

// Simple counter animation
function animateNumber(element, target) {
  const start = 0;
  const duration = 400;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const currentVal = (start + (target - start) * progress).toFixed(1);
    element.textContent = currentVal;

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = target.toLocaleString();
    }
  }

  requestAnimationFrame(update);
}
