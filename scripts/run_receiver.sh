#!/usr/bin/env bash

source .venv/bin/activate
python3 src/receiver/rx_stable.py "$@"

