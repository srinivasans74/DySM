#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export MPLBACKEND=Agg
./3runexperiment_adhoc.sh
./3runexperiment.sh
python3 figures.py
