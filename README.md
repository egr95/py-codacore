# py-codacore

A self-contained, up-to-date implementation of [CoDaCoRe](https://www.biorxiv.org/content/10.1101/2021.02.11.430695v1.full.pdf), in python, by the original authors.

For an equivalent implementation in the R programming language, check [R-codacore](https://github.com/egr95/R-codacore). If you are interested in reproducing the results in the [original paper](https://www.biorxiv.org/content/10.1101/2021.02.11.430695v1), check [this repo](https://github.com/cunningham-lab/codacore).

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions regarding the installation, usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>. 
Contributions and fixes are also welcome -- please create an issue, submit a pull request, or email me.

## How to run CoDaCoRe

1. To install codacore:

```bash
git clone https://github.com/egr95/py-codacore.git
cd py-codacore/
pip install .
```

2. To fit codacore on some data:
```python
from codacore.model import CodaCore
from codacore.datasets import simulate_hts
x, y = simulate_hts(1000, 100)
model = CodaCore(objective='binary_classification', type='balance')
model.fit(x, y)
model.summary()
```
### Unsupervised learning

Coming soon... If you would like access to an early version, get [in touch](mailto:eg2912@columbia.edu).

### Multi-omics

Coming soon... If you would like access to an early version, get [in touch](mailto:eg2912@columbia.edu).

