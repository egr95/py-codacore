import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="codacore",
    version="0.0.1",
    author="Elliott Gordon-Rodriguez",
    author_email="elliott.gordon.rodriguez@gmail.com",
    description="Learning Sparse Log-Ratios for High-Throughput Sequencing Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/egr95/py-codacore",
    python_requires='==3.5, ==3.6, ==3.7, ==3.8',
    install_requires=[
        'tensorflow>=2.4.0',
        'scikit-learn',
        'statsmodels',
    ],
    packages=['codacore'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
