import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from smolvlm_convert import export_all, export_rkllm

if __name__ == "__main__":
    export_all()
    export_rkllm()
