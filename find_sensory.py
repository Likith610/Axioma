import os
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))

neurons = pd.read_csv(
    os.path.join(_BASE, "neurons.csv"),
    dtype={"Root ID": "int64"},
    usecols=["Root ID", "Class", "Function", "Body Part", "Super Class", "Primary Cell Type"],
)
conn = pd.read_csv(
    os.path.join(_BASE, "connections_princeton.csv"),
    dtype={"pre_root_id": "int64"},
    usecols=["pre_root_id", "syn_count"],
)
out = conn.groupby("pre_root_id").syn_count.sum().rename("out_syn")
deg = conn.groupby("pre_root_id").size().rename("out_deg")

m = neurons.merge(out, left_on="Root ID", right_index=True).merge(
    deg, left_on="Root ID", right_index=True
)
sens = m[
    m["Class"]
    .fillna("")
    .str.contains(
        "olfactory_receptor|antennal|photoreceptor|mechano|auditory|gustatory|chordotonal|bristle",
        case=False,
        na=False,
    )
]

for cls, g in sens.groupby("Class"):
    g2 = g.sort_values("out_syn", ascending=False)
    top = g2.iloc[0]
    print(
        f"{cls}: {len(g):>5} neurons | top {top['Root ID']} "
        f"out_syn={top['out_syn']} type={top['Primary Cell Type']}"
    )

print()
top = sens.sort_values("out_syn", ascending=False).head(12)
print(
    top[
        ["Root ID", "Class", "Primary Cell Type", "Function", "out_syn", "out_deg"]
    ].to_string(index=False)
)
