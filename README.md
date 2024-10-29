### Unimatch: simulation tools for minimum-weight perfect-matching decoding on the surface code and color code

This repository contains the tools used to simulate the data in [https://arxiv.org/pdf/2306.16476](https://arxiv.org/pdf/2306.16476) and is subject to ongoing improvements.
The code borrows several tools and structures from David Tuckett's [`qecsim`](https://github.com/qecsim/qecsim), invokes [`pymatching`](https://github.com/oscarhiggott/PyMatching) as MWPM algorithm, 
and contains tools to implement the [unified color code decoder](https://arxiv.org/abs/2108.11395), and the [splitting method](https://arxiv.org/abs/1308.6270) for the simulation of logical error rates in the low
physical error rate regime.
