#!/usr/bin/env python3
"""Simple script to run the domain adaptation demo."""

import subprocess
import sys
import os


def main():
    """Run the Streamlit demo."""
    demo_path = os.path.join(os.path.dirname(__file__), "demo", "streamlit_demo.py")
    
    if not os.path.exists(demo_path):
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    print("Starting Domain Adaptation Demo...")
    print("The demo will open in your browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", demo_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
