"""Package critical data files into a zip for GitHub Releases (Option 2)."""
import zipfile, os

files = [
    "neurons.csv",
    "connections_princeton.csv",
    "active_neurons.csv",
]

with zipfile.ZipFile("axioma-data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f, arcname=f)
        else:
            print(f"WARNING: {f} not found — skip")

print("Created axioma-data.zip — upload this to GitHub Releases")
