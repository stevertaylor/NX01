NX01
====

Python code to perform isotropic, anisotropic (non-evolving/evolving)
searches for gravitational-waves via pulsar-timing.

[![DOI](https://zenodo.org/badge/20913/stevertaylor/NX01.svg)](https://zenodo.org/badge/latestdoi/20913/stevertaylor/NX01)

If you use this code then please [cite it](https://zenodo.org/badge/latestdoi/20913/stevertaylor/NX01). 

Perhaps you want the [webpage](http://stevertaylor.github.io/NX01/)?

## Contributors

* Justin Ellis
* Rutger van Haasteren
* Paul Baker
* Arian Azin
* Patrick Dean Mullen
* Mark Dewing
* Daniel George
* Miguel Holgado
* Michael Katolik
* Wei-Ting Liao
* Patrick Mullen
* Kedar Phadke

## Code list

* **NX01_master.py**: performs a full evolving-anisotropy GWB and
  noise analysis. Uses MultiNest or parallel-tempering sampling.
* **NX01_singlePsr.py**: performs a stochastic
  search for single-pulsar noise parameters within the reduced-rank
  time-frequency approximation. Uses MultiNest or parallel-tempering sampling.
* **NX01_AnisCoefficients.py**: utility file to create power-anisotropy
  basis-functions.
* **NX01_utils.py**: utility file.
* **NX01_processResults.py**: plotting script, adapted and extended from PAL.
* **NX01_psr.py**: utility file which defines the pulsar class for
storing all relevant variables.
* **NX01_datafile.py**: creates an hdf5 container to store all the
information in the pulsar class. Useful for storing array products.
* **NX01_jitter.pxy**: cython code to perform Sherman-Morrison
  block noise-matrix inversions when handling ECORR (jitter). 
* **NX01_bayesutils.py**: utilities file for generating plotting data.

## Getting things installed (from scratch)

1. Make sure you have a good python installation with a comprehensive
   collection of popular modules. I suggest
   [Anaconda](https://www.continuum.io/downloads), which has a large
   set of modules that we need. If you don't want to mess with your
   existing python installation, then I suggest you proceed with the
   following steps, resolving module errors as they are encountered by
   either finding the relevant module git repository or trying `pip
   install [...]`.
2. Download a recent version of
   [tempo2](https://bitbucket.org/psrsoft/tempo2). The download
   website has quite a good installation guide. During the
   installation process, copy T2runtime to a
   location of your choice. When you set the TEMPO2 environment
   variable, make sure to also add it to your bashrc file, and reload
   your bashrc file (`source ~/.bashrc`) to reflect the environment
   change.
3. After installing `tempo2`, you may also need to add the `tempo2`
   libraries to your library path. Do so by adding `export
   LD_LIBRARY_PATH=$TEMPO2/lib` to your bashrc file, and reload the
   bashrc file to reflect your changes.
4. Install [libstempo](https://github.com/vallis/libstempo), which is
   our python interface to tempo2. You can clone the repository (`git
   clone https://github.com/vallis/libstempo.git`) to wherever you
   want to store software on your machine, and install with `python
   setup.py install --with-tempo2=$TEMPO2`. If needed, more complete
   instructions can be found at `http://vallis.github.io/libstempo/`.
5. For NANOGrav analyses, download the
   [dataset](http://data.nanograv.org/) and follow the instructions in
   README.clock to update your tempo2 clock files. Note that the
   instructions tell you to copy `gbt2gps.dat` but the file is
   actually
   `gbt2gps.clk`. [If you have a recent tempo2 download (after November 2016) then manual clock updates are no longer needed]
6. Install [PTMCMCSampler](https://github.com/jellis18/PTMCMCSampler),
   which is the main sampler we use to
   explore the signal and noise parameter space. As before, clone with
   `git clone https://github.com/jellis18/PTMCMCSampler.git` and
   install with `python setup.py install`.
7. Install the python `ephem` package by executing `conda install
ephem`. Install the python `basemap` package by executing `conda
install basemap`.
8. Download [NX01](https://github.com/stevertaylor/NX01.git) by
   cloning the repository -- `git clone
   https://github.com/stevertaylor/NX01.git`.
9. Copy `nanograv-pulsar-store.ipynb` to a new notebook for your own
   specific use. Change all paths to reflect where you have NANOGrav
   par and tim data. NANOGrav noise files are included in the data
   directory. The end result should produce a directory
   containing `hdf5` files for all NANOGrav pulsars in the dataset.
10. Copy `PsrListings_GWB.txt` to a new file for your own specific
    use. Edit the paths to the `hdf5` files, `par` files, and `tim`
    files to reflect where you have stored them on your machine.
11. Execute `python NX01_master.py --help` to see all of the options
    available to you. These are quite extensive, and are being
    actively updated and improved.
12. To run an analysis which will recover the upper limit on the
    dimensionless strain amplitude which matches that reported in the
    [NANOGrav 9-year limit paper](http://arxiv.org/abs/1508.03024),
    execute `python NX01_master.py --from-h5
    --psrlist=./PsrListings_GWB.txt --nmodes=15 --incGWB --fix_slope
    --psrEndIndex=18 --dirExt=./chains_firsttests/`.
13. If you get errors with importing `NX01_jitter` then you may need
    to cythonize this module first. On the command line type `python
    setup-cython.py build_ext --inplace`. This should build the
    module, allowing you to repeat (12) without errors.
14. This will initiate a run, producing a parent directory
    `chains_firsttests`, and run-specific sub-directory.  This
    sub-directory will contain MCMC sample files and many auxillary
    files, amongst which is `parameter_list.txt`. This contains two
    columns; the first is a list of indices corresponding to
    columns of `chain_1.txt` where one can find the parameters of the
    second column.
15. After identifying the column of `chain_1.txt` which corresponds to
    `Agwb`, you can get an x% upper limit by executing
    the following:
		    
			import NX01_bayesutils as bu
			upper_lim = bu.confinterval(Agwb_samples, sigma=x/100, onesided=True)[1]
16. If you want to make a few summary plots, you can use
`NX01_processResults.py`. Execute `python NX01_processResults.py
--help` to see the options available to you. An example command is
`python NX01_processResults.py
--parentpath=/home/user/NX01/chains_firsttests --chaindir=nanograv_gwbdetect_noCorr_gam4p33_reddetectpowerlaw_nmodes15`
17. [IN DEVELOPMENT] If you want to make use of the NX01 GUI, you will need to install ipython widgets as follows: `conda install ipywidgets`.
18. [IN DEVELOPMENT] Open `NX01_GUI.ipynb` and execute the cells. This will produce an
    interactive GUI, allowing you to check boxes or enter options for
    your model. Clicking `Store Model` will create a json file called
    `mymodel.json` in the NX01 directory. Clicking `Engage` will begin
    running the analysis with output piped into the notebook
    cell. Alternatively you can take the json file and upload to your
    NX01 installation on a cluster, running with `python
    NX01_master.py --jsonModel=mymodel.json`.


## Single-pulsar noise analysis

It should be straightforward to perform a single-pulsar noise analysis
out of the box. 

Run `python NX01_singlePsr.py --help` for a list of all options. You
can direct NX01 to your particular pulsar using either (i) `parfile`
and `timfile` command-line arguments (full path needed); (ii) a full
path for `psrlist`, and `psrStartIndex` and `psrEndIndex` commands to
let the code know which position in the list it should select
(pythonic indexing is used); or
finally (iii) `from-h5` with a full
path for `psrlist`, and `psrStartIndex` and `psrEndIndex` commands to
let the code know which position in the list it should select.

An example run command would be:
```
python NX01_master.py
--parfile=./NANOGrav_9y/par/J1713+0747_NANOGrav_9yv1.t2.gls.strip.par
--timfile=./NANOGrav_9y/tim/J1713+0747_NANOGrav_9yv1.tim
--varyWhite --nmodes=20 --redSpecModel=powerlaw --redPrior=loguniform
--sampler=ptmcmc --dirExt=./nx01_tests/
```
You can set the sampler to PTMCMC (`ptmcmc`), MultiNest (`mnest`), or
PolyChord (`pchord`).

If you have MPI installed you can parallelise by running the
following:
```
mpirun -np 4 NX01_master.py --parfile=./NANOGrav_9y/par/J1713+0747_NANOGrav_9yv1.t2.gls.strip.par
--timfile=./NANOGrav_9y/tim/J1713+0747_NANOGrav_9yv1.tim
--varyWhite --nmodes=20 --redSpecModel=powerlaw --redPrior=loguniform
--sampler=ptmcmc --dirExt=./nx01_tests/
```
where 4 cores will produce 4 temperature chains in the
parallel-tempering MCMC sampling process. 


## Gravitational-wave searches

It is recommended to read in pulsars from their respective hdf5 files,
which you should have previously produced from the
`nanograv-pulsar-store.ipynb` notebook.

Run `python NX01_master.py --help` for a list of all options.

An example run command would be:
```
python NX01_master.py --from-h5
--psrlist=./PsrListings_GWB.txt --psrEndIndex=18 --nmodes=15 --redSpecModel=powerlaw --redPrior=uniform
--incGWB --fix_slope --gwbPrior=uniform --psrEndIndex=18
--sampler=ptmcmc --dirExt=./nx01_tests/
```
which will perform a GW background upper-limit analysis (without
correlations...to include correlations add `--incCorr`) with PTMCMC on the 18 pulsars analyzed in the 9-year NANOGrav limit paper.
