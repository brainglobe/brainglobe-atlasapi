pip install -e .[dev]
conda info -a
black ./ -l 79 --target-version py37 --check
pytest --cov=brainatlas_api