pip install -e .
pip install black
conda info -a
black ./ -l 79 --target-version py37 --check
pytest --cov=brainatlas_api