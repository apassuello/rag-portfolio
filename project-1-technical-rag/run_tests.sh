#!/bin/bash
# Test runner script with clean output
# Usage: ./run_tests.sh [pytest arguments]

# Set Python warnings environment to suppress external library warnings
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning,ignore::RuntimeWarning"

# Run pytest with all arguments passed through
python -m pytest "$@"