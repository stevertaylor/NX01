NX01
====

Python code to perform isotropic, anisotropic (non-evolving/evolving)
searches for gravitational-waves via pulsar-timing.

## Code list

* **NX01_master.py**: performs a full evolving-anisotropy GWB and
  noise analysis on the EPTAv2 dataset.
* **NX01_master_auto.py**: performs a full evolving-anisotropy GWB and
  noise analysis on the EPTAv2 dataset, but uses only the
  autocorrelation terms which are sufficient for upper-limits but not
  detection runs.
* **NX01_FixedNoise.py**: performs an evolving-anisotropy GWB analysis
  of the EPTAv2 dataset with pulsar noise values fixed at ML values
  obtained from single-pulsar analysis.
* **NX01_singlePsr_tf.py**: performs a PSO-maximisation or stochastic
  search for single-pulsar noise parameters within the reduced-rank
  time-frequency approximation.
* **NX01_freqStat.py**: computes the time-domain optimal-statistic for
  amplitude limits, and also the generalised anisotropic
  optimal-statistic for l>0.
* **NX01_noiseSims.py**: simulates .par and .tim files with noise
  characteristics equivalent to ML noise values obtained from
  single-pulsar analysis on real dataset. Background is injected with
  user-specified parameters.

* **NX01_AnisCoefficients.py**: utility file to create power-anisotropy
  basis-functions.
* **NX01_utils.py**: utility file.
* **NX01_plot.py**: plotting package, adapted and extended from PAL.
* **NX01_psr.py**: utility file which defines the pulsar class for
  storing all relevant variables.
* **NX01_bayesutils.py**: utilities file for generating plotting data.

* **NX01_master_EPTAsims.py**: specifically designed for EPTA
  injection and calibration tests. Only searches for GWB and
  red-noise.
* **NX01_singlePsr_tf_EPTAsims.py**: specifically designed for EPTA
  injection and calibration tests. Only searches for red-noise.

