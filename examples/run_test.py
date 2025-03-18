#!/usr/bin/env python3
import os
import sys
import subprocess
import torch

# Get PyTorch library path
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
print(f"PyTorch library path: {torch_lib_path}")

# Add PyTorch library path to LD_LIBRARY_PATH
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now try to import the module
try:
    import fastloader
    print("Successfully imported fastloader")
    
    # Create a test instance
    prefetcher = fastloader.Prefetcher()
    print("Successfully created Prefetcher instance")
    
    # Print information about available functions
    print("\nAvailable methods in Prefetcher class:")
    for attr in dir(prefetcher):
        if not attr.startswith('_'):  # Skip private methods
            print(f"  - {attr}")
            
    print("\nTest complete!")
except ImportError as e:
    print(f"Error importing fastloader: {e}")
    
    # Output additional diagnostic information
    print("\nDiagnostic information:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    
    # List files in current directory
    print("\nFiles in the current directory:")
    subprocess.run(["ls", "-la"], check=False)
    
    # Find the .so file
    print("\nSearching for fastloader.so files:")
    subprocess.run(["find", "..", "-name", "fastloader*.so"], check=False)
    
    # Check library dependencies
    so_files = subprocess.run(
        ["find", ".", "-name", "fastloader*.so"], 
        capture_output=True, 
        text=True,
        check=False
    ).stdout.strip().split('\n')
    
    if so_files:
        for so_file in so_files:
            if so_file:
                print(f"\nChecking dependencies for {so_file}:")
                subprocess.run(["ldd", so_file], check=False)
    
    sys.exit(1) 