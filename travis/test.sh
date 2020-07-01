pip install -r requirements.txt --no-cache-dir

pip install pytest
pip install pytest-cov

conda info -a
pytest --runslow --cov=bg_atlasapi
