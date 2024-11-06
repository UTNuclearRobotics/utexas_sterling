# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Python modules
pip install -e .

echo "Python dependencies installed and virtual environment set up."
echo "To enter the virtual environment, run: venv/bin/activate"
echo "To exit the virtual environment, run: deactivate"

deactivate
