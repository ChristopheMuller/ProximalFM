"""CLI entry point: python -m proximalfm.prior

Generates synthetic prior datasets for TabICL training.
"""

import runpy

runpy.run_module("proximalfm.prior._genload", run_name="__main__", alter_sys=True)
