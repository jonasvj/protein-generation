#!/bin/sh
# Run data scripts
src/data/download_raw_data.py
src/data/tidy_raw_data.py
src/data/create_datasets.py
