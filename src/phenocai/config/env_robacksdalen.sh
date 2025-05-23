#!/bin/bash

# PhenoCAI Environment Configuration for Röbäcksdalen

# Station Information
export PHENOCAI_CURRENT_STATION="robacksdalen"
export PHENOCAI_CURRENT_INSTRUMENT="RBD_AGR_PL01_PHE01"
export PHENOCAI_CURRENT_YEAR="2024"

# Station Details
export PHENOCAI_STATION_FULL_NAME="Röbäcksdalen"
export PHENOCAI_STATION_CODE="RBD"
export PHENOCAI_STATION_LAT="63.8111"
export PHENOCAI_STATION_LON="20.2394"

# Load base configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

echo "Loaded Röbäcksdalen configuration"
