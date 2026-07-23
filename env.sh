# Activate the virtual environment
source .venv/bin/activate

# Set the PYTHONPATH to include all the modules under `src`
_path="$PWD/src"
if [[ ":$PYTHONPATH:" != *":$_path:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$_path"
fi

# Add the `bin` directory to the PATH
_path="$PWD/bin"
if [[ ":$PATH:" != *":$_path:"* ]]; then
    export PATH="$_path:${PATH:+$PATH}"
fi
