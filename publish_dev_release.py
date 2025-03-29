#!/usr/bin/env python
"""
Script to build and publish the development release of sam_annotator to PyPI.
"""

import os
import subprocess
import sys
import shutil
import time
import requests

# Set your PyPI credentials here
# WARNING: Be careful not to share this file or commit it to a public repository
PYPI_TOKEN = "your_pypi_token_here"  # Replace with your actual PyPI token
TEST_PYPI_TOKEN = "pypi-AgENdGVzdC5weXBpLm9yZwIkM2E3YzczOGMtYWY2NS00ZTdmLWFlMDYtZTgwZWU4Y2RiYTE5AAIqWzMsIjg5OTgxZDllLWU1ZDAtNDc3ZC04YWJiLTE1MDc0OTY5MjEwMiJdAAAGIP5MJaBxIGTxnW7NMVA72HCJfZyVWuar--xU6aDIT1aL"  # Replace with your actual TestPyPI token

def run_command(command, description, env=None, exit_on_error=True):
    """Run a shell command and print its output."""
    print(f"\n\033[1;34m{description}...\033[0m")
    result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"\033[1;31mError output:\033[0m\n{result.stderr}")
    
    if result.returncode != 0:
        print(f"\033[1;31mCommand failed with return code {result.returncode}\033[0m")
        if exit_on_error:
            sys.exit(1)
        return False
    
    return True

def verify_package_upload(package_name, test_pypi=True):
    """Verify if a package has been successfully uploaded to PyPI/TestPyPI."""
    base_url = "https://test.pypi.org/pypi/" if test_pypi else "https://pypi.org/pypi/"
    url = f"{base_url}{package_name}/json"
    
    print(f"\n\033[1;34mVerifying package upload to {'TestPyPI' if test_pypi else 'PyPI'}...\033[0m")
    
    # Try multiple times with delays to allow for propagation
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                versions = list(data.get('releases', {}).keys())
                if versions:
                    print(f"\033[1;32mPackage found! Available versions: {', '.join(versions)}\033[0m")
                    return True
                else:
                    print(f"\033[1;33mPackage found but no releases available yet.\033[0m")
            else:
                print(f"\033[1;33mAttempt {attempt+1}: Package not found (Status code: {response.status_code})\033[0m")
        except Exception as e:
            print(f"\033[1;33mAttempt {attempt+1}: Error checking package: {e}\033[0m")
        
        if attempt < 2:  # Don't sleep on the last attempt
            print("Waiting 10 seconds before trying again...")
            time.sleep(10)
    
    print(f"\033[1;33mCould not verify package upload. It might still be processing.\033[0m")
    return False

def main():
    """Main function to build and publish the package."""
    # Prepare environment with credentials
    env = os.environ.copy()
    
    # Check if required packages are installed - Skip pip upgrade as it causes issues
    print("\n\033[1;34mChecking for required packages...\033[0m")
    try:
        # Install required packages for building and uploading
        run_command("pip install setuptools wheel twine requests", "Installing setuptools, wheel, twine and requests")
    except Exception as e:
        print(f"Error installing required packages: {e}")
        sys.exit(1)
    
    # Clean up previous builds
    print("\n\033[1;34mCleaning up previous builds...\033[0m")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    for item in os.listdir("."):
        if item.endswith(".egg-info"):
            shutil.rmtree(item)
    
    # Build the package using setup.py directly instead of build module
    run_command("python setup.py sdist bdist_wheel", "Building the package")
    
    # Show the built files
    run_command("ls -la dist/", "Files in dist directory")

    # Check for unusually large files
    dist_files = os.listdir("dist")
    for file in dist_files:
        file_path = os.path.join("dist", file)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        if file_size_mb > 100:  # Warn if any file is larger than 100MB
            print(f"\n\033[1;33mWarning: File {file} is very large ({file_size_mb:.2f} MB).\033[0m")
            print("This may cause upload issues. Consider reducing the package size.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Upload aborted.")
                return
    
    # Ask for confirmation before uploading
    print("\n\033[1;33mReady to upload to PyPI.\033[0m")
    print("\033[1;33mThis will publish your development release publicly.\033[0m")
    
    choice = input("Do you want to continue? (y/n): ").strip().lower()
    if choice != 'y':
        print("Upload aborted.")
        return
    
    # Ask which repository to use
    print("\n\033[1;33mWhere would you like to publish?\033[0m")
    print("1. TestPyPI (for testing)")
    print("2. PyPI (production)")
    
    repo_choice = input("Enter 1 or 2: ").strip()
    
    if repo_choice == '1':
        # Create .pypirc content for TestPyPI
        with open(os.path.expanduser("~/.pypirc"), "w") as f:
            f.write(f"""[distutils]
index-servers =
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = {TEST_PYPI_TOKEN}
""")
        # Upload to TestPyPI with retry mechanism
        upload_success = False
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"\n\033[1;34mAttempt {attempt+1}/{max_retries} to upload to TestPyPI...\033[0m")
                
                # Set a longer timeout for the upload
                result = subprocess.run(
                    "python -m twine upload --repository testpypi dist/*",
                    shell=True, capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    upload_success = True
                    break
                else:
                    print(f"\033[1;31mUpload failed with return code {result.returncode}\033[0m")
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(f"\033[1;31mError output:\033[0m\n{result.stderr}")
                    
                    # Check if it's just a connection error
                    if "ConnectionError" in result.stderr or "RemoteDisconnected" in result.stderr:
                        print("\033[1;33mConnection error detected. This might be a network issue.\033[0m")
                    
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)  # Increase wait time with each retry
                        print(f"\033[1;33mWaiting {wait_time} seconds before retrying...\033[0m")
                        time.sleep(wait_time)
            except Exception as e:
                print(f"\033[1;31mException during upload: {e}\033[0m")
                if attempt < max_retries - 1:
                    print(f"\033[1;33mWaiting 10 seconds before retrying...\033[0m")
                    time.sleep(10)
        
        # Verify the upload even if we had errors
        package_name = "sam-annotator"  # Adjust if your package name is different
        verify_package_upload(package_name, test_pypi=True)
        
        print("\n\033[1;32mIf your package is visible on TestPyPI, you can install it with:\033[0m")
        print("\033[1;33mpip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam-annotator\033[0m")
    else:
        # Create .pypirc content for PyPI
        with open(os.path.expanduser("~/.pypirc"), "w") as f:
            f.write(f"""[distutils]
index-servers =
    pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = {PYPI_TOKEN}
""")
        # Upload to PyPI with retry mechanism
        upload_success = False
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"\n\033[1;34mAttempt {attempt+1}/{max_retries} to upload to PyPI...\033[0m")
                
                result = subprocess.run(
                    "python -m twine upload dist/*",
                    shell=True, capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    upload_success = True
                    break
                else:
                    print(f"\033[1;31mUpload failed with return code {result.returncode}\033[0m")
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(f"\033[1;31mError output:\033[0m\n{result.stderr}")
                    
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        print(f"\033[1;33mWaiting {wait_time} seconds before retrying...\033[0m")
                        time.sleep(wait_time)
            except Exception as e:
                print(f"\033[1;31mException during upload: {e}\033[0m")
                if attempt < max_retries - 1:
                    print(f"\033[1;33mWaiting 10 seconds before retrying...\033[0m")
                    time.sleep(10)
        
        # Verify the upload even if we had errors
        package_name = "sam-annotator"  # Adjust if your package name is different
        verify_package_upload(package_name, test_pypi=False)
        
        print("\n\033[1;32mIf your package is visible on PyPI, you can install it with:\033[0m")
        print("\033[1;33mpip install --pre sam-annotator\033[0m")
    
    # Clean up credentials file
    try:
        os.remove(os.path.expanduser("~/.pypirc"))
        print("\nTemporary credentials file removed.")
    except:
        print("\nWarning: Could not remove temporary credentials file.")

if __name__ == "__main__":
    main() 