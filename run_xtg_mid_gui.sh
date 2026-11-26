#!/usr/bin/env bash
# Launch the XTG MID analysis GUI on Maxwell / EuXFEL machines.
#
#   ./run_xtg_mid_gui.sh --proposal 11456                  # real data, nothing loaded yet (creates the venv on first use)
#   ./run_xtg_mid_gui.sh --proposal 7016 --scans 90-100    # load these runs at start
#   ./run_xtg_mid_gui.sh --backend simulated --scans 101-149   # fake data, to check the display
#
# The venv lives next to this folder (../xtg_mid_env) unless XTG_MID_ENV is set, e.g.
#   XTG_MID_ENV=/gpfs/exfel/exp/MID/<cycle>/p<proposal>/usr/Software/xtg_mid_env ./run_xtg_mid_gui.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${XTG_MID_ENV:-$(dirname "$HERE")/xtg_mid_env}"

# 1) EuXFEL python (numpy, h5py, extra_data, Qt) if the module system exists
if command -v module >/dev/null 2>&1 || [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load exfel exfel-python 2>/dev/null || echo "warning: 'module load exfel exfel-python' failed, using $(command -v python3)"
fi

# 2) venv on top of it (first run only; needs internet, i.e. Maxwell, not the online cluster)
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "creating venv $ENV_DIR"
    python3 -m venv --system-site-packages "$ENV_DIR"
    "$ENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    "$ENV_DIR/bin/python" -m pip install --quiet -e "$HERE"
    "$ENV_DIR/bin/python" -c "import pyqtgraph.Qt" 2>/dev/null \
        || "$ENV_DIR/bin/python" -m pip install --quiet PyQt5
fi

# 3) display check
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "error: no display. Use FastX (max-display.desy.de) or 'ssh -X', or a control-room machine." >&2
    exit 1
fi

export XTG_MID_START_DIR="$PWD"     # Sources_11456.yaml / Sources_7016.yaml are looked for here first
cd "$HERE"
exec "$ENV_DIR/bin/python" -m xtg_mid_gui "$@"
