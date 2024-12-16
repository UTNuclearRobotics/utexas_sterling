sudo apt update

# Install apt dependencies
while read -r package; do
    sudo apt install -y "$package"
done < apt_packages.txt

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip3 install -r requirements.txt

# Install Python modules
pip3 install -e .

echo "Python dependencies installed and virtual environment set up."
echo "To enter the virtual environment, run: venv/bin/activate"
echo "To exit the virtual environment, run: deactivate"

deactivate
