from os import path

from setuptools import find_namespace_packages, setup

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "bg-space >= 0.5.0",
    "click",
    "meshio",
    "numpy",
    "pandas",
    "requests",
    "rich >= 9.0.0",
    "tifffile",
    "treelib",
]


setup(
    name="bg-atlasapi",
    version="1.0.2",
    description="A lightweight python module to interact with "
    "atlases for systems neuroscience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage>=5.0.3",
            "bump2version",
            "pre-commit",
            "flake8",
            "check-manifest",
        ]
    },
    python_requires=">=3.6",
    entry_points={"console_scripts": ["brainglobe = bg_atlasapi.cli:bg_cli"]},
    packages=find_namespace_packages(exclude=("docs", "tests*")),
    include_package_data=True,
    url="https://brainglobe.info/atlas-api",
    project_urls={
        "Source Code": "https://github.com/brainglobe/bg-atlasapi",
        "Bug Tracker": "https://github.com/brainglobe/bg-atlasapi/issues",
        "Documentation": "https://docs.brainglobe.info/bg-atlasapi",
    },
    author="Luigi Petrucco, Federico Claudi, Adam Tyson",
    author_email="code@adamltyson.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
