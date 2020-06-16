from setuptools import setup, find_namespace_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="brainatlas-api",
    version="0.0.2rc0",
    description="A lightweight python module to interact with atlases for systems neuroscience",
    install_requires=requirements,
    extras_require={
        "dev": [
            "allensdk",
            "sphinx",
            "brainio>=0.0.16",
            "vtkplotter",
            "recommonmark",
            "sphinx_rtd_theme",
            "pydoc-markdown",
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage",
            "pre-commit",
            "PyMCubes",
            "bgspace",
        ]
    },
    python_requires=">=3.6, <3.8",
    entry_points={
        "console_scripts": [
            "brainatlas_config = brainatlas_api.config:cli_modify_config",
            "brainatlas_list_atlases = brainatlas_api.list_atlases:list_atlases",
        ]
    },
    packages=find_namespace_packages(exclude=("atlas_gen", "docs", "tests*")),
    include_package_data=True,
    url="https://github.com/brainglobe/brainatlas-api",
    author="Luigi Petrucco, Federico Claudi, Adam Tyson",
    author_email="adam.tyson@ucl.ac.uk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
