
pip install black
pip install flake8

flake8
black ./ -l 79 --target-version py37 --check
