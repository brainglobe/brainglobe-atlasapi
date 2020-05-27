wget $MINICONDA_URL -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda info -a
conda create -n test-environment python=$TRAVIS_PYTHON_VERSION
source activate test-environment
