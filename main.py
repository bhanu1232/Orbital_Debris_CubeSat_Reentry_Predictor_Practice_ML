"""
AstroDecay ML - FastAPI Service
Orbital Debris & CubeSat Reentry Lifespan Predictor Backend
"""

import os
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
import uvicorn
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="AstroDecay ML",
    description="Atmospheric Drag & Orbital Debris Reentry Lifespan Predictor",
    version="1.0.0"
)

# Load trained model artifacts
MODEL_PATH = "orbit_model.joblib"
if not os.path.exists(MODEL_PATH):
    print("Model artifact not found. Generating now...")
    from train_model import train_and_export_models
    train_and_export_models()

model_bundle: Dict[str, Any] = joblib.load(MODEL_PATH)
reg_pipeline: Any = model_bundle["reg_pipeline"]
cls_pipeline: Any = model_bundle["cls_pipeline"]
model_metrics: Dict[str, Any] = model_bundle["metrics"]
feature_importances: Dict[str, float] = model_bundle["feature_importances"]

# Request Schema (Pydantic v2 compliant)
class OrbitalPredictionRequest(BaseModel):
    object_type: str = Field(default="CubeSat (3U/6U)", description="Object classification category")
    perigee_km: float = Field(default=380.0, ge=200.0, le=1200.0, description="Perigee Altitude in km")
    apogee_km: float = Field(default=410.0, ge=200.0, le=2000.0, description="Apogee Altitude in km")
    inclination_deg: float = Field(default=51.6, ge=0.0, le=180.0, description="Orbital Inclination in degrees")
    ballistic_coeff: float = Field(default=45.0, ge=5.0, le=350.0, description="Ballistic Coefficient in kg/m^2")
    solar_flux_f107: float = Field(default=140.0, ge=60.0, le=300.0, description="Solar Radio Flux Index F10.7 in sfu")
    rcs_m2: float = Field(default=0.08, ge=0.01, le=50.0, description="Radar Cross Section in m^2")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object_type": "CubeSat (3U/6U)",
                    "perigee_km": 380.0,
                    "apogee_km": 410.0,
                    "inclination_deg": 51.6,
                    "ballistic_coeff": 45.0,
                    "solar_flux_f107": 140.0,
                    "rcs_m2": 0.08
                }
            ]
        }
    }

# Realistic Mission Presets for instant testing
PRESETS: Dict[str, Dict[str, Any]] = {
    "cubesat_low": {
        "name": "3U University CubeSat (Low LEO)",
        "object_type": "CubeSat (3U/6U)",
        "perigee_km": 380.0,
        "apogee_km": 405.0,
        "inclination_deg": 51.6,
        "ballistic_coeff": 45.0,
        "solar_flux_f107": 140.0,
        "rcs_m2": 0.08,
        "description": "Standard 3U CubeSat deployed from ISS orbit, passive decay without propulsion."
    },
    "starlink_derelict": {
        "name": "Decommissioned Starlink Bus",
        "object_type": "Decommissioned Constellation",
        "perigee_km": 540.0,
        "apogee_km": 550.0,
        "inclination_deg": 53.0,
        "ballistic_coeff": 85.0,
        "solar_flux_f107": 175.0,
        "rcs_m2": 1.8,
        "description": "Megaconstellation flat-panel satellite with active deorbit failure."
    },
    "rocket_stage": {
        "name": "Heavy Spent Upper Stage",
        "object_type": "Rocket Upper Stage",
        "perigee_km": 720.0,
        "apogee_km": 760.0,
        "inclination_deg": 71.0,
        "ballistic_coeff": 180.0,
        "solar_flux_f107": 95.0,
        "rcs_m2": 9.5,
        "description": "Dense rocket body in higher LEO during low solar cycle; centuries-long orbital risk."
    },
    "solar_max_debris": {
        "name": "Solar Storm Fragment (High Drag)",
        "object_type": "Fragmentation Debris",
        "perigee_km": 310.0,
        "apogee_km": 340.0,
        "inclination_deg": 98.2,
        "ballistic_coeff": 22.0,
        "solar_flux_f107": 225.0,
        "rcs_m2": 0.35,
        "description": "Lightweight solar panel debris during peak geomagnetic solar maximum."
    }
}

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "online",
        "model_loaded": True,
        "metrics": model_metrics
    }

@app.get("/api/presets")
def get_presets() -> Dict[str, Dict[str, Any]]:
    return PRESETS

@app.get("/api/model-info")
def get_model_info() -> Dict[str, Any]:
    return {
        "metrics": model_metrics,
        "feature_importances": feature_importances,
        "supported_types": [
            "CubeSat (3U/6U)",
            "MicroSat (50-100kg)",
            "Rocket Upper Stage",
            "Fragmentation Debris",
            "Decommissioned Constellation"
        ]
    }

@app.post("/api/predict")
def predict_orbital_reentry(req: OrbitalPredictionRequest) -> Dict[str, Any]:
    # Altitude validation
    if req.apogee_km < req.perigee_km:
        raise HTTPException(
            status_code=400, 
            detail="Apogee altitude must be greater than or equal to perigee altitude."
        )

    # Prepare input dataframe
    input_data = pd.DataFrame([{
        "object_type": req.object_type,
        "perigee_km": req.perigee_km,
        "apogee_km": req.apogee_km,
        "inclination_deg": req.inclination_deg,
        "ballistic_coeff": req.ballistic_coeff,
        "solar_flux_f107": req.solar_flux_f107,
        "rcs_m2": req.rcs_m2
    }])

    # Predict lifespan in log scale then convert back
    pred_log_days = float(reg_pipeline.predict(input_data)[0])
    pred_days = float(np.expm1(pred_log_days))
    pred_days = max(1.0, round(pred_days, 1))

    # Predict risk class
    pred_risk = str(cls_pipeline.predict(input_data)[0])
    
    # Calculate years
    years = round(pred_days / 365.25, 2)

    # Estimate daily altitude decay rate
    daily_decay_meters = round(float(np.clip((350.0 / (pred_days ** 0.85)) * (req.solar_flux_f107 / 120.0), 0.01, 200.0)), 2)

    # FCC 5-year compliance (Rule: Satellites in LEO must deorbit within 5 years of mission end)
    fcc_compliant = bool(pred_days <= (5 * 365.25))

    # Status classification badges
    if pred_days < 90:
        status_color = "#ff3d71"  # Critical red
        status_badge = "CRITICAL DEORBIT IMMINENT"
        action = "Reentry track activated. Trajectory monitoring within standard atmospheric capture window."
    elif pred_days < 365:
        status_color = "#ffaa00"  # Amber warning
        status_badge = "HIGH DRAG / PASSIVE DECAY"
        action = "Rapid altitude loss regime. Object will naturally clear LEO corridor in under 12 months."
    elif pred_days <= 1825:
        status_color = "#00e5ff"  # Cyan compliant
        status_badge = "FCC 5-YEAR COMPLIANT"
        action = "Within international sustainable orbit regulations. Expected safe burn-up within 5 years."
    else:
        status_color = "#7b68ee"  # Purple long-term
        status_badge = "PERSISTENT ORBITAL DEBRIS"
        action = "Exceeds 5-year natural de-orbit window. Active Debris Removal (ADR) or propulsion required."

    return {
        "status": "success",
        "prediction": {
            "lifespan_days": pred_days,
            "lifespan_years": years,
            "risk_category": pred_risk,
            "daily_decay_rate_meters": daily_decay_meters,
            "fcc_5yr_compliant": fcc_compliant,
            "status_badge": status_badge,
            "status_color": status_color,
            "action_recommendation": action
        },
        "telemetry_summary": {
            "mean_altitude_km": round((req.perigee_km + req.apogee_km) / 2.0, 1),
            "eccentricity": round((req.apogee_km - req.perigee_km) / (req.perigee_km + req.apogee_km + 12742.0), 4),
            "solar_activity_regime": "High Storm" if req.solar_flux_f107 > 180 else ("Moderate" if req.solar_flux_f107 > 110 else "Solar Minimum")
        }
    }

# Mount static files folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home() -> FileResponse:
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
