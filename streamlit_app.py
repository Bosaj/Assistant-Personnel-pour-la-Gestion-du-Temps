import os
import sys
import runpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(CURRENT_DIR, "dashboard")

sys.path.insert(0, os.path.join(CURRENT_DIR, "notebooks"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "scripts"))
sys.path.insert(0, DASHBOARD_DIR)

runpy.run_path(os.path.join(DASHBOARD_DIR, "app.py"), run_name="__main__")
