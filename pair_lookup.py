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
    ],
)

def show(title, df):
    print(f"== {title} ==")
    sub = df[df["Soma side"].notna()].copy()
    for side in ["left", "right"]:
        g = sub[sub["Soma side"].astype(str).str.contains(side, case=False)]
        print(f"  {side}: {len(g)} neurons | sample: " + ", ".join(
            f"{r['Primary Cell Type']}#{r['Root ID']}" for _, r in g.head(3).iterrows()
        ))
    print()

bristle = neurons[neurons["Class"].fillna("").str.contains("bristle", case=False)]
show("bristle (touch)", bristle)

orn = neurons[neurons["Class"].fillna("").str.contains("olfactory_receptor", case=False)]
show("ORN (smell)", orn)

neck = neurons[neurons["Class"].fillna("").str.contains("neck_motor", case=False)]
show("neck motor (turn)", neck)

leg = neurons[neurons["Class"].fillna("").str.contains("leg_motor", case=False)]
show("leg motor (walk)", leg)

wing = neurons[neurons["Class"].fillna("").str.contains("wing_motor", case=False)]
show("wing motor (fly)", wing)

photo = neurons[neurons["Class"].fillna("").str.contains("photoreceptor", case=False)]
show("photoreceptor (vision)", photo)
