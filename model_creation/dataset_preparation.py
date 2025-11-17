import numpy as np
from pycaret.regression import *
import pandas as pd

data = pd.read_csv("vehicle-dataset-from-cardekho/car details v4.csv")

# convert data to normal
data['Price'] /= 81.28      # INR -> USD

data['Engine'] = data['Engine'].str.extract('(\d+)').astype(float)

data['Max Power (hp)'] = pd.to_numeric(data['Max Power'].str.extract(
    '(\d+\.?\d*)')[0], errors='coerce') * 1.01387  # bhp -> hp
data['Max Power (rpm)'] = pd.to_numeric(
    data['Max Power'].str.extract('@\s*(\d+)')[0], errors='coerce')

data['Max Torque (Nm)'] = pd.to_numeric(
    data['Max Torque'].str.extract('(\d+\.?\d*)')[0], errors='coerce')
data['Max Torque (rpm)'] = pd.to_numeric(
    data['Max Torque'].str.extract('@\s*(\d+)')[0], errors='coerce')

# convert owner to numeric, NaN for unknown
data['Owner'].replace('UnRegistered Car', np.nan, inplace=True)
owner_mapping = {
    'First': 1,
    'Second': 2,
    'Third': 3,
    'Fourth': 4,
    '4 or More': 5
}
data['Owner'] = data['Owner'].map(owner_mapping)

# rename columns to corect names
data.rename(
    columns={
        'Make': 'Brand',
        'Price': 'Price (USD)',
        'Kilometer': 'Mileage (km)',
        'Engine': 'Engine Capacity (cc)'
    },
    inplace=True)

# delete useless columns
data.drop(
    columns=[
        'Location',
        'Length',
        'Width',
        'Height',
        'Max Power',
        'Max Torque'
    ],
    inplace=True)
