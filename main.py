from enum import Enum
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pycaret.regression import load_model, predict_model
import pandas as pd
from pydantic import BaseModel, field_validator

app = FastAPI(title="Car Price Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = load_model("deployment")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class CarFeatures(BaseModel):
    brand: str | None = None
    model_name: str | None = None
    year: int | None = None
    mileage: float | None = None
    fuel_type: str | None = None
    transmission: str | None = None
    color: str | None = None
    seating_capacity: int | None = None
    seller_type: str | None = None
    owner: int | None = None
    drivetrain: str | None = None
    engine_capacity: float | None = None
    fuel_tank_capacity: float | None = None
    max_power_hp: float | None = None
    max_power_rpm: int | None = None
    max_torque_Nm: float | None = None
    max_torque_rpm: int | None = None

    @field_validator('*', mode='before')
    @classmethod
    def empty_str_to_none(cls, v, info):
        """Converts any incoming empty string "" into None before Pydantic validation."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return
        return v


class FuelType(str, Enum):
    petrol = "Petrol"
    diesel = "Diesel"
    electric = "Electric"
    cng = "CNG"
    lpg = "LPG"


class Transmission(str, Enum):
    manual = "Manual"
    automatic = "Automatic"


class SellerType(str, Enum):
    individual = "Individual"
    corporate = "Corporate"


class Drivetrain(str, Enum):
    fwd = "FWD"
    rwd = "RWD"
    awd = "AWD"


def form_body_dependency(
    brand: str | None = Form("", description="Brand of the car"),
    model_name: str | None = Form("", alias="model",
                                  description="Name of the car"),
    year: int | str | None = Form(
        "", description="Manufacturing year of the car"),
    mileage: float | str | None = Form(
        "", description="Total kilometers driven"),
    fuel_type: FuelType | str | None = Form(
        "", description="Fuel type of the car"),
    transmission: Transmission | str | None = Form("",
                                                   description="Gear transmission of the car"),
    color: str | None = Form("", description="Color of the car"),
    seating_capacity: int | str | None = Form("",
                                              description="Maximum people that can fit in the car"),
    seller_type: SellerType | str | None = Form("",
                                                description="Tells if car is sold by individual or dealer"),
    owner: int | str | None = Form("",
                                   description="Number of previous owners"),
    drivetrain: Drivetrain | str | None = Form("",
                                               description="Type of drivetrain of the car"),
    engine_capacity: float | str | None = Form("",
                                               description="Engine capacity of the car in cc"),
    fuel_tank_capacity: float | str | None = Form("",
                                                  description="Maximum fuel capacity of the car in litres"),
    max_power_hp: float | str | None = Form(
        "", description="Maximum power of the car in hp"),
    max_power_rpm: float | str | None = Form(
        "", description="Number of rpm at maximum power"),
    max_torque_Nm: float | str | None = Form(
        "", description="Maximum torque of the car in Nm"),
    max_torque_rpm: float | str | None = Form(
        "", description="Number of rpm at maximum torque"),
) -> CarFeatures:
    # Use locals() to capture all arguments passed to this function
    return CarFeatures(**locals())


class PredictedPrice(BaseModel):
    predicted_price: float


@app.post("/predict", response_model=PredictedPrice)
def predict_price(features: CarFeatures = Depends(form_body_dependency)):
    """Predict the price of the car in USD based on parameters"""

    # Mapping Pydantic fields to DataFrame column names efficiently
    df_columns_map = {
        "brand": "Brand",
        "model_name": "Model",
        "year": "Year",
        "mileage": "Mileage (km)",
        "fuel_type": "Fuel Type",
        "transmission": "Transmission",
        "color": "Color",
        "seating_capacity": "Seating Capacity",
        "seller_type": "Seller Type",
        "owner": "Owner",
        "drivetrain": "Drivetrain",
        "engine_capacity": "Engine Capacity (cc)",
        "fuel_tank_capacity": "Fuel Tank Capacity",
        "max_power_hp": "Max Power (hp)",
        "max_power_rpm": "Max Power (rpm)",
        "max_torque_Nm": "Max Torque (Nm)",
        "max_torque_rpm": "Max Torque (rpm)",
    }

    # Create input dictionary using dictionary comprehension
    input_dict = {
        col_name: getattr(features, field_name)
        for field_name, col_name in df_columns_map.items()
    }

    input_data = pd.DataFrame([input_dict])
    # print(input_data)
    prediction = predict_model(model, input_data)['prediction_label'][0]

    return {"predicted_price": round(float(prediction), 2)}
