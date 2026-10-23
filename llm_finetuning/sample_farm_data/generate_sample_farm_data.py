import pandas as pd
import numpy as np

np.random.seed(42) # For reproducibility
num_samples = 2000

crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut", "Mustard", "Pearl Millet", "Sorghum", "Finger Millet", "Chickpea", "Pigeon Pea", "Potato", "Tomato", "Onion", "Chili", "Banana", "Tea", "Sunflower"]
stages = ["Germination", "Vegetative", "Flowering", "Yield Formation", "Ripening"]
soil_types = ["Sandy", "Loam", "Clay", "Black Soil"]

print("Generating 3000 samples...")
data = {
    # crop features
    "crop": np.random.choice(crops, num_samples),
    "growth_stage": np.random.choice(stages, num_samples),
    "disease_detected": np.random.choice(["Yes", "No", "No", "No"], num_samples),
    
    # soil features
    "soil_type": np.random.choice(soil_types, num_samples),
    "soil_moisture_pct": np.random.randint(10, 85, num_samples),
    "npk_nitrogen_mg_kg": np.random.randint(10, 400, num_samples),
    "npk_phosphorus_mg_kg": np.random.randint(5, 250, num_samples),
    "npk_potassium_mg_kg": np.random.randint(10, 580, num_samples),

    # weather features
    "dht11_humidity_pct": np.random.randint(20, 94, num_samples),
    "temp_c": np.random.randint(10, 45, num_samples),
    "forecast_rain_mm": np.random.choice([0, 0, 0, 0, 2, 8, 18, 35, 75], num_samples), # Heavy weight on 0mm
    "wind_speed_kmh": np.random.randint(3, 45, num_samples),
    "solar_irradiance_wm2": np.random.randint(100, 1001, num_samples),
    "atmospheric_pressure_hpa": np.random.randint(990, 1021, num_samples)
}
print("Samples generated")

print("Writing samples to: sample_farm_data/numerical_farm_scenarios.csv")
df = pd.DataFrame(data)
df.to_csv("llm_finetuning/sample_farm_data/numerical_farm_scenarios.csv", index=False)
print(f"{num_samples} samples saved!")
