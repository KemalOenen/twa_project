#!/bin/bash

set -e

PARAMETERS_FILE="twa/parameters.py"

PHASE_MEANS=(
    "0.0"
    "1.5707963267948966"
    "3.141592653589793"
)

PHASE_STDS=(
    "0.0"
    "0.3490658503988659"
    "0.6981317007977318"
    "1.0471975511965976"
    "1.3962634015954636"
    "1.7453292519943295"
    "2.0943951023931953"
    "2.443460952792061"
    "2.792526803190927"
    "3.141592653589793"
)

# Save the original parameters file so it can always be restored.
cp "$PARAMETERS_FILE" "${PARAMETERS_FILE}.backup"

restore_parameters() {
    mv "${PARAMETERS_FILE}.backup" "$PARAMETERS_FILE"
}

trap restore_parameters EXIT

for phase_b in "${PHASE_MEANS[@]}"; do
    for phase_std in "${PHASE_STDS[@]}"; do

        echo "Running phase_b = $phase_b, sigma_phi = $phase_std"

        sed -i.tmp \
            -E "s/^([[:space:]]*)phase_b_rad=.*/\1phase_b_rad=${phase_b},/" \
            "$PARAMETERS_FILE"

        sed -i.tmp \
            -E "s/^([[:space:]]*)relative_phase_std_rad=.*/\1relative_phase_std_rad=${phase_std},/" \
            "$PARAMETERS_FILE"

        rm -f "${PARAMETERS_FILE}.tmp"

        python run_simulation.py
    done
done
