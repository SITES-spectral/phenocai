#!/bin/bash

# PhenoCAI Environment Configuration for Lönnstorp

# Station Information
export PHENOCAI_CURRENT_STATION="lonnstorp"
export PHENOCAI_CURRENT_INSTRUMENT="LON_AGR_PL01_PHE01"
export PHENOCAI_CURRENT_YEAR="2024"

# Station Details
export PHENOCAI_STATION_FULL_NAME="Lönnstorp"
export PHENOCAI_STATION_CODE="LON"
export PHENOCAI_STATION_LAT="55.6686"
export PHENOCAI_STATION_LON="13.1073"

# Load base configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

echo "Loaded Lönnstorp configuration"
