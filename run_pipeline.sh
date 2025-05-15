#!/bin/bash

set -e  # Exit on any error

echo "Running DVC pipeline..."
dvc repro

echo "Adding changes to Git..."
git add dvc.yaml dvc.lock data/processed/*.csv.dvc

echo "Committing..."
git commit -m "Run preprocessing and update processed data"

echo "Pushing to Git..."
git push

echo "Pushing data to DVC remote..."
dvc push

echo "Done."