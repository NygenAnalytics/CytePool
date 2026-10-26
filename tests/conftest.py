"""Settings shared by the whole test suite."""

import os

# Figures are drawn without windows, as on a machine without a display. Where a
# display is available, matplotlib chooses an interactive backend, whose Tk objects
# raise errors when the garbage collector finalizes them on another thread.
os.environ.setdefault("MPLBACKEND", "Agg")
