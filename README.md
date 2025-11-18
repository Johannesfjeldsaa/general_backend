# general backend for my projects
* setup logging
* masking of files
* saving utils
* safe xarray operations
* ...

# Setting up a virtual environment and installation

You can create a virtual environment either in the current directory or with a custom prefix path:

```bash
# Create virtual environment in current directory
python -m venv venv
# Activate virtual environment
source venv/bin/activate

# OR

# Create virtual environment with custom path
python -m venv /path/to/your/envs/project_backend_env

source /path/to/your/envs/project_backend_env/bin/activate
```

Install the package in development mode (with dev dependencies)

```bash
pip install -e .[dev]
```

For a regular installation:

```bash
pip install .
```

The package dependencies are managed through `pyproject.toml` and will be automatically installed.

# Dev
Before commiting run dev dependecies:
```bash
# Run quality checks
black --check .          # Check formatting
isort --check-only .     # Check import sorting
flake8 .                 # Check style/errors
mypy src/                # Type checking
pytest                   # Run tests
```
or as a one-liner
```bash
black --check . && isort --check-only . && flake8 . && mypy src/ && echo "✅ All code quality checks passed!"
```