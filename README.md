# polya-gamma
Efficiently sample from the Polya-Gamma distribution using NumPy's Generator interface.
![densities](./scripts/densities.svg)

## Dependencies
- Numpy >= 1.17 


## Installation
```shell
$ pip install -U polyagamma
```


## Example

### Python

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

# one can choose a sampling method from {devroye, alternate, gamma}.
# If not given, the default behaviour is a hybrid sampler that picks
# the best method based on the parameter values
out = g.polyagamma(method="devroye")

# other numpy distributions are still accessible
print(g.standard_normal())
print(g.standard_gamma())
```
### C
For an example of how to use `polyagamma` in a C program, see [here][1].


## TODO
- ~~Add devroye and gamma convolution methods.~~
- ~~Add the "alternate" sampling method.~~
- ~~Add the hybrid sampler based on a combination of available methods.~~
- ~~Add array broadcasting support for paramater inputs.~~
- Add the "saddle point approximation" method.


## References
- Luc Devroye. "On exact simulation algorithms for some distributions related to Jacobi theta functions." Statistics & Probability Letters, Volume 79, Issue 21, (2009): 2251-2259.
- Polson, Nicholas G., James G. Scott, and Jesse Windle. "Bayesian inference for logistic models using Pólya–Gamma latent variables." Journal of the American statistical Association 108.504 (2013): 1339-1349.
- J. Windle, N. G. Polson, and J. G. Scott. "Improved Polya-gamma sampling". Technical Report, University of Texas at Austin, 2013b.
- Windle, Jesse, Nicholas G. Polson, and James G. Scott. "Sampling Polya-Gamma random variates: alternate and approximate techniques." arXiv preprint arXiv:1405.0506 (2014)


[1]: ./examples/c_polyagamma.c
