#!/usr/bin/env python
"""
Simplified script to build and publish the development release of sam_annotator to PyPI.
"""

import os
import subprocess
import sys
import shutil

# Set your PyPI credentials here
# WARNING: Be careful not to share this file or commit it to a public repository
PYPI_TOKEN = "your_pypi_token_here"  # Replace with your actual PyPI token
TEST_PYPI_TOKEN = "pypi-AgENdGVzdC5weXBpLm9yZwIkM2E3YzczOGMtYWY2NS00ZTdmLWFlMDYtZTgwZWU4Y2RiYTE5AAIqWzMsIjg5OTgxZDllLWU1ZDAtNDc3ZC04YWJiLTE1MDc0OTY5MjEwMiJdAAAGIP5MJaBxIGTxnW7NMVA72HCJfZyVWuar--xU6aDIT1aL"

def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n\033[1;34m{description}...\033[0m")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print output in real-time
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        
        if stdout_line:
            print(stdout_line.rstrip())
        if stderr_line:
            print(f"\033[1;31m{stderr_line.rstrip()}\033[0m")
            
        if not stdout_line and not stderr_line and process.poll() is not None:
            break
    
    retcode = process.poll()
    if retcode != 0:
        print(f"\033[1;31mCommand failed with return code {retcode}\033[0m")
        return False
    
    return True

def clean_build_folders():
    """Clean up previous build artifacts."""
    print("\n\033[1;34mCleaning up previous builds...\033[0m")
    folders_to_remove = ['build', 'dist', '*.egg-info']
    
    for folder in folders_to_remove:
        if '*' in folder:
            # Handle wildcard pattern
            import glob
            for path in glob.glob(folder):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed {path}")
        elif os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed {folder}")

def create_pypirc(is_test=False):
    """Create a .pypirc file with the appropriate credentials."""
    token = TEST_PYPI_TOKEN if is_test else PYPI_TOKEN
    repo = "testpypi" if is_test else "pypi"
    repo_url = "https://test.pypi.org/legacy/" if is_test else "https://upload.pypi.org/legacy/"
    
    pypirc_content = f"""[distutils]
index-servers =
    {repo}

[{repo}]
repository = {repo_url}
username = __token__
password = {token}
"""
    
    pypirc_path = os.path.expanduser("~/.pypirc")
    with open(pypirc_path, "w") as f:
        f.write(pypirc_content)
    
    return pypirc_path

def main():
    """Main function to build and publish the package."""
    # 1. Check and install dependencies
    if not run_command("python -m pip install --upgrade pip setuptools wheel twine", 
                     "Installing/upgrading required packages"):
        return
    
    # 2. Clean up previous builds
    clean_build_folders()
    
    # 3. Build the package manually using setup.py
    if not run_command("python setup.py sdist bdist_wheel", 
                     "Building the package"):
        return
    
    # 4. Show the built files
    run_command("ls -la dist/", "Files in dist directory")
    
    # 5. Ask for confirmation before uploading
    print("\n\033[1;33mReady to upload to PyPI.\033[0m")
    print("\033[1;33mThis will publish your development release publicly.\033[0m")
    
    choice = input("Do you want to continue? (y/n): ").strip().lower()
    if choice != 'y':
        print("Upload aborted.")
        return
    
    # 6. Ask which repository to use
    print("\n\033[1;33mWhere would you like to publish?\033[0m")
    print("1. TestPyPI (for testing)")
    print("2. PyPI (production)")
    
    repo_choice = input("Enter 1 or 2: ").strip()
    is_test = repo_choice == '1'
    
    # 7. Create .pypirc file
    pypirc_path = create_pypirc(is_test)
    
    # 8. Upload to the selected repository
    repo_name = "testpypi" if is_test else "pypi"
    if run_command(f"python -m twine upload --repository {repo_name} dist/*",
                 f"Uploading to {'TestPyPI' if is_test else 'PyPI'}"):
        
        print(f"\n\033[1;32mSuccess! Package uploaded to {'TestPyPI' if is_test else 'PyPI'}.\033[0m")
        print("\033[1;32mTo install, run:\033[0m")
        
        if is_test:
            print("\033[1;33mpip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam_annotator\033[0m")
        else:
            print("\033[1;33mpip install --pre sam_annotator\033[0m")
    
    # 9. Clean up the credentials file
    try:
        os.remove(pypirc_path)
        print("\nTemporary credentials file removed.")
    except Exception as e:
        print(f"\nWarning: Could not remove temporary credentials file: {e}")

if __name__ == "__main__":
    main() 