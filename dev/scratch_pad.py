import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.circle_packing.openevolve_output.best.best_program import run_packing, visualize

centers, radii, sum_radii = run_packing()
print(f"Sum of radii: {sum_radii}")
visualize(centers, radii)