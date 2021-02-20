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
    install_requires=[
        'tensorflow>=2.1.0',
        'scikit-learn',
        'statsmodels',
    ]
    packages=['codacore'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
