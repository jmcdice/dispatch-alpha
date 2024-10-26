#!/usr/bin/env bash

#!/usr/bin/env bash

source env.sh
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 src/receiver/rx_stable.py "$@"

