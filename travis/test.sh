pip install -r requirements.txt --no-cache-dir

pip install black
pip install pytest

conda info -a
black ./ -l 79 --target-version py37 --check
pytest --cov=brainatlas_api