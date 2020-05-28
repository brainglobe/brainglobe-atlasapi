from setuptools import setup, find_namespace_packages

requirements = ["tqdm", "numpy", "tifffile", "treelib", "pandas"]


setup(
    name="brainatlas-api",
    version="0.0.1c",
    description="A lightweight python module to interact with atlases for systems neuroscience",
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
        ]
    },
    python_requires=">=3.6, <3.8",
    packages=find_namespace_packages(exclude=("docs", "tests*")),
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
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
