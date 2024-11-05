setup_venv() {
    # Create and activate Python virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install Python dependencies
    pip install -r requirements.txt

    echo "Python dependencies installed and virtual environment set up."
    echo "To enter the virtual environment, run: venv/bin/activate"
    echo "To exit the virtual environment, run: deactivate"
    
    deactivate
}

enter_venv() {
    source /opt/ros/humble/setup.bash
    source venv/bin/activate
}

case "$1" in
    setup)
        setup_venv
        ;;
    enter)
        enter_venv
        ;;
    *)
        echo "Usage: $0 {setup|enter}"
        exit 1
        ;;
esac