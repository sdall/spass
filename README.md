# Discovering Significant Patterns under Sequential False Discovery Control

This repository provides a Julia library that implements the *Significant Pattern Association* (Spass) algorithm. By leveraging a binomial redundancy test for a sequentially-updating maximum entropy null-model, Spass provides an efficient method for discovering concise sets of statistically significantly non-redundant higher-order feature interactions (i.e., patterns). To highlight commonalities and differences between groups, Spass statistically associates each pattern with a subset of groups.

The code is a from-scratch implementation of algorithms described in the [paper](https://doi.org/10.1145/3534678.3539398). 

```
Sebastian Dalleiger and Jilles Vreeken. 2022. 
Discovering Significant Patterns under Sequential False Discovery Control. 
(KDD '22), pp. 263–272. https://doi.org/10.1145/3534678.3539398
```

Please consider [citing](CITATION.bib) the paper.

[Contributions](CONTRIBUTING.md) are welcome.

## Installation

To install the library from the REPL:
```julia-repl
julia> using Pkg; Pkg.add(url="https://github.com/sdall/spass.git")
```

To install the library from the command line:
```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/sdall/spass.git")'
```

To set up the command line interface (CLI) located in `bin/spass.jl`:

1. Clone the repository:
```sh
git clone https://github.com/sdall/spass
```
2. Install the required dependencies including the library:
```sh
julia -e 'using Pkg; Pkg.add(path="./spass"); Pkg.add.(["Comonicon", "CSV", "GZip", "JSON"])'
```

## Usage

A typical usage of the library is: 
```julia-repl
julia> using Spass: spass, FDR, FWER, patterns
julia> p = spass(FWER, X; alpha = 0.01)
julia> patterns(p)
```
For more information, please see the documentation:
```julia-repl
help?> spass
```

A typical usage of the command line interface is:
```sh
chmod +x bin/spass.jl
bin/spass.jl dataset.dat.gz dataset.labels.gz --alpha=0.01 --fdr > output.json
```
The output contains `patterns` and `executiontime` in seconds (cf. `--measure-time` for details).
For more information regarding usage, additional options, or input format, please see the provided documentation:
```sh
bin/spass.jl --help
```
