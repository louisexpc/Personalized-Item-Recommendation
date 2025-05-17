#!/bin/bash

if ! command -v python &> /dev/null
then
    echo "python could not be found, trying python3"
    if ! command -v python3 &> /dev/null
    then
        echo "python3 could not be found either. Please install Python."
        exit 1
    fi
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Running main.py using ${PYTHON_CMD}..."
${PYTHON_CMD} main.py "$@"

echo "Script finished."