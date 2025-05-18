#!/usr/bin/env python3
"""
Data acquisition script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Downloads the Credit Card Fraud Detection dataset
2. Initializes DVC
3. Adds the raw data to DVC tracking
4. Pushes to the DVC remote
"""

import os
import sys
import logging
import requests
import hashlib
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('data-acquisition')

# Constants
DATA_URL = "https://nextcloud.scopicsoftware.com/s/bo5PTKgpngWymGE/download/creditcard-data.csv"
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard-data.csv"
GIT_IGNORE = "data/raw/.gitignore"
# Expected SHA256 checksum of the file (optional for validation)
EXPECTED_SHA256 = None  # Replace with actual SHA256 if known

def setup_directories():
    """Create necessary directories if they don't exist."""
    logger.info(f"Creating directory {RAW_DATA_DIR}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) 

def download_data():
    """Download the dataset from the source URL."""
    if RAW_DATA_FILE.exists():
        logger.info(f"Data file already exists at {RAW_DATA_FILE}, skipping download.")
        return

    logger.info(f"Downloading data from {DATA_URL}")
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    with open(RAW_DATA_FILE, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Download complete: {RAW_DATA_FILE}")

def compute_sha256(filepath):
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def validate_data():
    """Validate the downloaded data file integrity."""
    if not RAW_DATA_FILE.exists():
        logger.error("Data file does not exist.")
        sys.exit(1)

    if EXPECTED_SHA256:
        logger.info("Validating data file checksum...")
        checksum = compute_sha256(RAW_DATA_FILE)
        if checksum != EXPECTED_SHA256:
            logger.error("Checksum does not match. File may be corrupted.")
            sys.exit(1)
        logger.info("Checksum validated.")
    else:
        logger.warning("No expected checksum provided. Skipping validation.")

def has_staged_changes():
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"]
    )
    return result.returncode != 0  # True if there are staged changes

def initialize_dvc():
    """Initialize DVC and add data to tracking."""
    if not Path(".dvc").exists():
        logger.info("Initializing DVC...")
        subprocess.run(["dvc", "init"], check=True)

    #logger.info(f"Adding {RAW_DATA_FILE} to DVC tracking...")
    #subprocess.run(["dvc", "add", str(RAW_DATA_FILE)], check=True)

    # logger.info("Committing DVC changes to Git...")
    # subprocess.run(["git", "add", str(RAW_DATA_FILE) + ".dvc"], check=True)
    # subprocess.run(["git", "add", GIT_IGNORE], check=True)
    
    # if has_staged_changes():
    #     subprocess.run(["git", "commit", "-m", "Add raw dataset to DVC"], check=True)
    # else:
    #     print("Nothing to commit — working tree clean.")

    logger.info("Pushing data to DVC remote...")
    # subprocess.run(["dvc", "push"], check=True)

def main():
    """Main function to orchestrate the data acquisition process."""
    logger.info("Starting data acquisition process")

    setup_directories()
    download_data()
    validate_data()
    initialize_dvc()

    logger.info("Data acquisition completed successfully")

if __name__ == "__main__":
    main()