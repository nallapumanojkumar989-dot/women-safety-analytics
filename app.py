import os
import sys

# Entry point wrapper for root execution
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "women-safety-analytics")
sys.path.insert(0, project_dir)
os.chdir(project_dir)

import app
