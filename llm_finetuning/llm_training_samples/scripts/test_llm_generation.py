import pandas as pd

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

NVIDIA_API_KEY = "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"

teacher_llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4.1-flash",
    api_key=NVIDIA_API_KEY,
    temperature=0.4,
    max_completion_tokens=262144
)

teacher_prompt = PromptTemplate(
    template="""
        You are an experienced argultural assistant having emence domain knowledge. You are given the real time farm (crop and soil) and weather data. You task is to summarize the data in simple words then suggest the farmer a set of instructions they must follow today.\n
        Realtime farm data: 
        - Crop: {crop},
        - Growth Stage: {growth_stage},
        - Pathogen/Disease Detected: {disease_detected},
        - Soil Type: {soil_type},
        - Soil Moisture: {soil_moisture_pct}%,
        - Nitrogen (N): {npk_nitrogen_mg_kg} mg/kg,
        - Phosphorus (P): {npk_phosphorus_mg_kg} mg/kg,
        - Potassium (K): {npk_potassium_mg_kg} mg/kg,
        - Ambient Temperature: {temp_c}°C,
        - Relative Humidity: {dht11_humidity_pct}%,
        - Atmospheric Pressure: {atmospheric_pressure_hpa} hPa,
        - Solar Irradiance: {solar_irradiance_wm2} W/m²,
        - Wind Speed: {wind_speed_kmh} km/h,
        - Forecasted Rain: {forecast_rain_mm} mm

        Note: Return only two plain paragraphs. First for summary. Second for instructions. No formatting or pointers needed.
    """,
    input_variables=["crop", "growth_stage", "disease_detected", "soil_type", "soil_moisture_pct", "npk_nitrogen_mg_kg", "npk_phosphorus_mg_kg", "npk_potassium_mg_kg", "temp_c", "dht11_humidity_pct", "atmospheric_pressure_hpa", "solar_irradiance_wm2", "wind_speed_kmh", "forecast_rain_mm"]
)

parser = StrOutputParser()

chain = teacher_prompt | teacher_llm | parser

farm_data = pd.read_csv("llm_finetuning/sample_farm_data/numerical_farm_scenarios.csv")

i = 123
curr_farm_data = {
    "crop": farm_data.loc[i, "crop"],
    "growth_stage": farm_data.loc[i, "growth_stage"],
    "disease_detected": farm_data.loc[i, "disease_detected"],
    "soil_type": farm_data.loc[i, "soil_type"],
    "soil_moisture_pct": farm_data.loc[i, "soil_moisture_pct"],
    "npk_nitrogen_mg_kg": farm_data.loc[i, "npk_nitrogen_mg_kg"],
    "npk_phosphorus_mg_kg": farm_data.loc[i, "npk_phosphorus_mg_kg"],
    "npk_potassium_mg_kg": farm_data.loc[i, "npk_potassium_mg_kg"],
    "temp_c": farm_data.loc[i, "temp_c"],
    "dht11_humidity_pct": farm_data.loc[i, "dht11_humidity_pct"],
    "atmospheric_pressure_hpa": farm_data.loc[i, "atmospheric_pressure_hpa"],
    "solar_irradiance_wm2": farm_data.loc[i, "solar_irradiance_wm2"],
    "wind_speed_kmh": farm_data.loc[i, "wind_speed_kmh"],
    "forecast_rain_mm": farm_data.loc[i, "forecast_rain_mm"]
}

teacher_response = chain.invoke(curr_farm_data)

print(teacher_response)
