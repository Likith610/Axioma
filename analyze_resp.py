import os
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))

resp = pd.read_csv(os.path.join(_BASE, "fly_response.csv"))
neurons = pd.read_csv(
    os.path.join(_BASE, "neurons.csv"),
    dtype={"Root ID": "int64"},
    usecols=["Root ID", "Class", "Soma side", "Primary Cell Type"],
)

m = resp.merge(neurons, left_on="neuron_id", right_on="Root ID")
m["side"] = m["Soma side"].fillna("").astype(str)
m["is_leg"] = m["Class"].fillna("").astype(str) == "leg_motor_neuron"
m["is_neck"] = m["Class"].fillna("").astype(str) == "neck_motor_neuron"

for sensor in ["smellL", "smellR", "touchL", "touchR"]:
    g = m[m["sensor"] == sensor]
    legs = g[g["is_leg"]]
    legL = legs[legs["side"].str.contains("left")]["signal"].sum()
    legR = legs[legs["side"].str.contains("right")]["signal"].sum()
    necks = g[g["is_neck"]]
    neckL = necks[necks["side"].str.contains("left")]["signal"].sum()
    neckR = necks[necks["side"].str.contains("right")]["signal"].sum()
    print(
        f"{sensor}: leg_left={legL:9.0f} leg_right={legR:9.0f}  "
        f"(L/R={legL/legR if legR else float('nan'):5.3f}) | "
        f"neck_left={neckL:7.0f} neck_right={neckR:7.0f}"
    )
