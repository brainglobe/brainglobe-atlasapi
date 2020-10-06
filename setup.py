from setuptools import setup, find_namespace_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_dev.txt") as f:
    requirements_dev = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="bg-atlasapi",
    version="1.0.0",
    description="A lightweight python module to interact with "
    "atlases for systems neuroscience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require=dict(dev=requirements_dev),
    python_requires=">=3.6",
    entry_points={"console_scripts": ["brainglobe = bg_atlasapi.cli:bg_cli"]},
    packages=find_namespace_packages(exclude=("docs", "tests*")),
    include_package_data=True,
    url="https://github.com/brainglobe/bg-atlasapi",
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
