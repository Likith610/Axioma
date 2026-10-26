import os
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))

neurons = pd.read_csv(
    os.path.join(_BASE, "neurons.csv"),
    dtype={"Root ID": "int64"},
    usecols=[
        "Root ID",
        "Class",
        "Primary Cell Type",
        "Function",
        "Soma side",
        "Body Part",
        "Nerve",
    ],
)
motors = neurons[
    neurons["Class"].fillna("").str.contains("motor", case=False, na=False)
]
print("Motor classes and how many neurons each:")
print(motors.groupby("Class").size().to_string())
print()
print("Motor types (Primary Cell Type) counts:")
print(
    motors.groupby(["Class", "Primary Cell Type"])
    .size()
    .sort_values(ascending=False)
    .head(40)
    .to_string()
)
