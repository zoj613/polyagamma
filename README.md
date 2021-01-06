# polya-gamma
Efficiently sample from the Polya-Gamma distribution using NumPy's Generator interface.


## Dependencies
- Numpy >= 1.17 


## Installation
```shell
$ pip install -U polyagamma
```


## Example
`polyagamma` can act as a drop-in replacement for numpy's Generator class.
```python
import numpy as np

from polyagamma import default_rng, Generator

g = Generator(np.random.PCG64())  # or use default_rng()
print(g.polyagamma())

# Get a 5 by 10 array of PG(1, 2) variates.
print(g.polyagamma(z=2, size=(5, 10)))

# Pass sequences as input. Numpy's broadcasting semantics apply here.
h = [[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]]
print(g.polyagamma(h, 1))

# Pass an output array
out = np.empty(5)
g.polyagamma(out=out)
print(out)

# other numpy distributions are still accessible
print(g.standard_normal())
print(g.standard_gamma())
```

## TODO
- ~~Add devroye and gamma convolution methods.~~
- Add the "alternative" sampling method.
- Add the "saddle point approximation" method.
- Add the hybrid sampler based on all four methods.
- ~~Add array broadcasting support for paramater inputs.~~

## References
- Polson, Nicholas G., James G. Scott, and Jesse Windle. "Bayesian inference for logistic models using Pólya–Gamma latent variables." Journal of the American statistical Association 108.504 (2013): 1339-1349.
- J. Windle, N. G. Polson, and J. G. Scott. "Improved Polya-gamma sampling". Technical Report, University of Texas at Austin, 2013b.
- Windle, Jesse, Nicholas G. Polson, and James G. Scott. "Sampling Polya-Gamma random variates: alternate and approximate techniques." arXiv preprint arXiv:1405.0506 (2014)
