from setuptools import setup, find_namespace_packages

requirements = ["tqdm", "numpy", "tifffile", "treelib", "pandas"]


setup(
    name="brainatlas-api",
    version="0.0.1b",
    description="TBC",
    install_requires=requirements,
    extras_require={
        "dev": [
            "allensdk",
            "sphinx",
            "recommonmark",
            "sphinx_rtd_theme",
            "pydoc-markdown",
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage",
            "pre-commit",
            "vtkplotter",
            "PyMCubes",
            "brainio",
        ]
    },
    python_requires=">=3.6",
    packages=find_namespace_packages(exclude=("docs", "tests*")),
    include_package_data=True,
    url="https://github.com/brainglobe/brainatlas-api",
    author="Luigi Petrucco, Federico Claudi, Adam Tyson",
    author_email="adam.tyson@ucl.ac.uk",
    classifiers=["Development Status :: 3 - Alpha"],
    zip_safe=False,
)
