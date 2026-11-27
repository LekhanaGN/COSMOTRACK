
import csv
import io
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"

GROUPS = [
    ("stations", "stations.csv", 5, "STATION"),
    ("starlink", "satellites.csv", 35, "SATELLITE"),
    ("fengyun-1c-debris", "debris.csv", 20, "DEBRIS"),
]


def fetch_catalog(group, filename, limit, object_type):
    filepath = DATA_DIR / filename

    # Use local cached data if available.
    if filepath.exists():
        print(f"Using cached {filename}")

        with filepath.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    print(f"Downloading {group} from CelesTrak...")

    response = requests.get(
        BASE_URL,
        params={
            "GROUP": group,
            "FORMAT": "CSV"
        },
        timeout=60
    )

    response.raise_for_status()

    reader = csv.DictReader(io.StringIO(response.text))

    required = {
        "OBJECT_NAME",
        "NORAD_CAT_ID",
        "EPOCH",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
    }

    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Invalid catalogue: {group}")

    rows = list(reader)

    # Select approximate LEO candidates.
    rows = [
        row for row in rows
        if float(row["MEAN_MOTION"]) > 11.25
    ]

    # Prioritise ISS in the station catalogue.
    if group == "stations":
        rows.sort(
            key=lambda row:
            row["NORAD_CAT_ID"] != "25544"
        )

    rows = rows[:limit]

    if not rows:
        raise ValueError(f"No orbital data found: {group}")

    for row in rows:
        row["COSMOTRACK_TYPE"] = object_type

    with filepath.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} objects to {filename}")

    return rows


def load_catalog():
    objects = {}

    for group, filename, limit, object_type in GROUPS:
        rows = fetch_catalog(
            group, filename, limit, object_type
        )

        for row in rows:
            objects[int(row["NORAD_CAT_ID"])] = row

    return list(objects.values())


if __name__ == "__main__":
    catalogue = load_catalog()

    print("\nCOSMOTRACK CATALOGUE")
    print("=" * 50)
    print(f"Total tracked objects: {len(catalogue)}")

    for row in catalogue[:10]:
        print(
            row["NORAD_CAT_ID"],
            row["OBJECT_NAME"],
            row["COSMOTRACK_TYPE"]
        )

    print("\nCatalogue ready!")
