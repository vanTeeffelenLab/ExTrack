import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open("extrack/version.py").read())

setuptools.setup(
    name="extrack",
    version=__version__,
    author="Francois Simon",
    author_email="simon.francois@protonmail.com",
    description="SPT kinetic modelling and states annotation of tracks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancoisSimon/ExTrack-python3",
    project_urls={
        "Bug Tracker": "https://github.com/FrancoisSimon/ExTrack-python3",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    install_requires=['lmfit', 'xmltodict', 'pandas', 'matplotlib'],
    python_requires=">=3.6",
)
