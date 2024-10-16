#!/usr/bin/env bash

source .venv/bin/activate
python3 src/transmitter/tx_stable.py "$@"
