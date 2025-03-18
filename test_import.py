import os
import sys
import subprocess

print("Python path:", sys.path)
print("\nTrying to import fastloader...")

try:
    # Add the current directory to Python path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Get the libtorch path from the environment
    torch_lib_path = subprocess.check_output(
        ["python", "-c", "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
        text=True
    ).strip()
    
    print(f"PyTorch library path: {torch_lib_path}")
    
    # Set the LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    
    # Try importing the module
    import fastloader
    print("Successfully imported fastloader")
    
    # Print module info
    print(dir(fastloader))
    
except Exception as e:
    print(f"Error: {e}")
    
    # Show shared libraries
    print("\nChecking dependencies:")
    shared_lib = "./fastloader.cpython-310-x86_64-linux-gnu.so"
    if os.path.exists(shared_lib):
        subprocess.run(["ldd", shared_lib])
    else:
        print(f"Library not found: {shared_lib}") 