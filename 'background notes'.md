As for the next step, I would suggest you take a look at different possible data sources. You have identified [Planck](https://www.cosmos.esa.int/web/planck/pla) and [WMAP](https://lambda.gsfc.nasa.gov/product/wmap/current/index.html). Please also take a look at NASA's [LAMBDA](https://lambda.gsfc.nasa.gov/product/), and see how it is related to the two sources. 

To give a more practical direction to this phase of your work, let's go ahead and download (a piece of) the data from each of these data sources and visualize them. You can use a Jupyter Notebook for this purpose.

--------------------------------------------------------------------
NASA'S LAMBDA is a long-term archive of data made publicly available for CMB researchers to use, giving access to cosmology missions, software tools, and links to other relevant sites. [Atacama (ACT)](https://act.princeton.edu/) will also be explored.

<b> Some reminders/vocabulary...</b>

- <b>$Ω$</b>: ratio of the universe's density $ρ$ to the critical density $ρ_{c}$ 
- <b>ΛCDM</b> (lambda cold dark matter): standard model of the big bang theory
    - three major components
        - cosmological constant (Λ) associated with dark energy
        - postulated cold dark matter (CDM)
        - ordinary matter
    - describes existence and structure of the CMD 
    - large scale structure of galaxy distribution
    - observed abundances of hydrogen (and deuterium), helium, and lithium
    - accelerating expansion of the universe obtained from redshift of distant objects
- <b>$\sigma$</b>: standard deviation/error
- <b>E-modes</b>: polarization of light associated with matter density in the CMB
    - perpendicular and parallel from direction of propogation
    - no curl, just divergence
- <b>B-modes</b>: polarization of light associated with gravitational influence in CMB
    - 45° rotation from direction of propogation
    - no divergence, just curl
    - currently being researched on how it can prove theory of inflation
- [more on light polarization and EM waves](https://science.nasa.gov/ems/02_anatomy/#:~:text=the%20image%20data.-,POLARIZATION,-One%20of%20the)

<b>Definitions for CMB Radiation Research</b>
- <b>Decoupling</b>: the first moment microwave photons could travel interruption-free of unbound electrons 
- <b>Surface of Last Scattering</b>: a "shell" or "surface" at the right distance in space where the photons from decoupling can be received

$$C(\theta) = \frac{l}{4\pi} \sum_{l=0}^\infty (2l+1)C_{l}P_{l}\cos(\theta)$$

- <b>Correlation Function $C(\theta)$</b>: statistical measure of total temperature fluctuation (via [Cambridge](https://people.ast.cam.ac.uk/~pettini/Intro%20Cosmology/Lecture10.pdf#page=3))
    - $C(\theta)$ compares the temperature fluctuations between two points on the SoLSc separated by an angle $\theta$
    - the result is an average of all the products between temperature fluctuations of a pair separated by $\theta$
    - <b>$l$</b>: multipole moment
    - <b>$C_{l}$</b>: the multipole components of the Legendre Polynomial expansion $P_{l}$
- <b>Angular Power Spectra</b>
    - $C_{l}^{XY, data} = \frac{l}{2l+1} \sum_{m=-l}^l a_{X, lm} a^{*}_{Y, lm}$
    - where X, Y can be T, E, or B
    - $a$ represents the spin-2 coefficient of a polarization mode
- <b>TT (temperature)</b>: spectrum data derived from the following values at each <i>$l$</i>
    - power spectrum: $\frac{l}{2\pi} \left(l+1\right) C_{l}^{TT, data}$ given in units of $\mu K^{2}$
    - noise: $\frac{l}{2\pi} \left(l+1\right) N_{l}$
    - effective sky fraction $f_{sky, l}$
    - best fit theory spectrum $C_{l}^{TT, th}$
- <b>TE</b>: temperature-E-mode cross-power spectrum (temperature-polarization spectrum)
- <b>EE</b>: E-mode polarization angular auto-power spectrum
- TB, EB, and BB are expected to be zero - in general, they can be non-zero from data contamination (foreground signals)

<b>Helpful Links</b>
- WMAP: [definitions/methodology](https://iopscience.iop.org/article/10.1088/0067-0049/192/2/16#apjs374861s2), [yr9 supplement](https://lambda.gsfc.nasa.gov/product/wmap/dr5/pub_papers/nineyear/supplement/WMAP_supplement.pdf)
- Atacama (ACT): [data products](https://lambda.gsfc.nasa.gov/product/act/actpol_prod_table.html#:~:text=The%20ACTPol%20EE%2C%20TT%20and%20TE%20power%20spectra)
- Planck: [legacy archive](http://pla.esac.esa.int/pla/#home), [explanatory supplement](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Main_Page)
- some [guidance](https://cocalc.com/share/public_paths/b5e7ec1aaacde391032512694af4e4be53af04c4/CMB_Analysis_Summer_School_PartOne-checkpoint.ipynb) on mapping the cmb
- if healpy works, see [this](https://healpy.readthedocs.io/en/latest/tutorial.html#Visualization) for more help on mapping cmb


<b>Questions for Nima</b>
- do I need healpy to make a cmb map?
    - if so, help with installing to environment?
- from ACT data products: total spectra vs cmb spectra 
- what data should i be looking for to find the temperature fluctuations?
    - the data products of act, wmap, and plank that I've looked at so far are all scaled differently
    - should I just manually calculate the fluctuations with the correlation function?
    - how do you know what angle to pick? is it just based on precision?
- i found a source that might help me figure out how to map it, is this the right direction?
    - if $C_l$ is the constant that goes with every term of $P_l$ expansion, what is $D_l$?
    - the multipole moments are for every term, but how would that relate to the map of the cmb? 
    - is it an axis on the map? or does each term have its own map of the cmb across every angle of correlation function?
------------------------
- do I need healpy to make a cmb map?
    - no, that's only for spherical maps, just looking for matrix-like data first
    - [use this wmap data](https://lambda.gsfc.nasa.gov/product/wmap/dr5/m_products.html#:~:text=I%20Maps%20per%20Differencing%20Assembly-,Temperature%20(I)%20maps,-for%20each%20D/A%20at%20HEALPix%20Nside%3D512)
        - can use astropy (do this way)
        - DS9 to plot fits files (as another option)
        - pick one fits file and load it and explore with it (click wget first)
            - K9 and Q9 or some letter and number at the end of the file name might mean the band filter?
        - just focus on wmap first
        - focus on given images, not taking the raw data to make an image
- from ACT data products: total spectra vs cmb spectra 
    - not what we're looking for
- what data should i be looking for to find the temperature fluctuations?
- i found a source that might help me figure out how to map it, is this the right direction?
    - some good information (first paragraph of section 1.1), but we don't want simulations

- goal with project is to build a model that can take raw cmb data and make it's own conclusions
- neural networks
- see paper sent on slack, project is based on this idea

fourier spectrum $D_l$ and angular power spectrum $C_l$
- wave vs time (time domain) -> fourier transform -> frequenncy on x-axis vs amplitude (aka frequency domain)
- autocorrelation -> ft -> power spectrum

$l$ is x-axis of power spectrum (aka frequency f)
- has to do more with distance
- specific to frequency but for cmb
- $C_l$ is the amplitude (power at each $l$)
- just think of $D_l$ as a "transformation" or "form" of $C_l$ -> important note on that [here](https://cocalc.com/share/public_paths/b5e7ec1aaacde391032512694af4e4be53af04c4/CMB_Analysis_Summer_School_PartOne-checkpoint.ipynb#:~:text=The%20correct%20thing,to%20seeing%20plotted.)