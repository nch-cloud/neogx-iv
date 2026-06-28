#!/usr/bin/env python3

from pathlib import Path
import requests

RECORD_ID = "20615809"


def destination_path(filename: str) -> Path:
    # data/
    if filename.startswith('cohort'):
        return Path("data") / filename

    if filename == "search-results.csv.gz":
        return Path("data") / filename

    if filename == "static_features.csv.gz":
        return Path("data/feature_matrices") / filename

    if filename.startswith("pheno_features-") and filename.endswith(".csv.gz"):
        return Path("data/feature_matrices") / filename

    # assets/
    if filename.endswith("_R_labels.csv"):
        return Path("assets/phecodes") / filename

    if filename.startswith("hpo_representative_map-") and filename.endswith(".json"):
        return Path("assets") / filename

    # Everything else goes into the HPO ontology directory
    return Path("assets/hpo-v2025-10-22") / filename


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dest}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def main():
    record = requests.get(
        f"https://zenodo.org/api/records/{RECORD_ID}"
    ).json()

    for file_info in record["files"]:
        filename = file_info["key"]
        url = file_info["links"]["self"]

        download_file(url, destination_path(filename))


if __name__ == "__main__":
    main()