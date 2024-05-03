import os
from re import search
import numpy as np
from astropy.io import fits
import astropy.visualization as vis
import matplotlib.pyplot as plt
import time
from IPython.display import display, HTML, Image, clear_output

class ViewFits:
    
    def FitsMapper(files, hdul_index, nrows, ncols, cmap, interpolation, 
                   animation=False, interval=None, stretch=None, 
                   vmin=None, vmax=None, contrast=None, bias=None, power=1, percentile=1):
        
        if animation == False:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 9))

        for i, file in enumerate(files):
            with fits.open(file) as hdul:
                data = hdul[hdul_index].data
            
            # creating dictionaries for available intervals and stretches in astropy.visualization (vis in this notebook)
            intervals = {'ZScale': vis.ZScaleInterval(),
                        'MinMax': vis.MinMaxInterval(),
                        'Percentile': vis.PercentileInterval(percentile),
                        'AsymPercentile': vis.AsymmetricPercentileInterval(vmin, vmax),
                        'Manual': vis.ManualInterval(vmin=vmin, vmax=vmax)}
            
            stretches = {'Linear': vis.LinearStretch(),
                        'Asinh': vis.AsinhStretch(),
                        'Log': vis.LogStretch(),
                        'Sqrt': vis.SqrtStretch(),
                        'Hist' : vis.HistEqStretch(data),
                        'Power': vis.PowerStretch(power),
                        'Sinh': vis.SinhStretch(),
                        'Contrast': vis.ContrastBiasStretch(contrast=contrast, bias=bias)}

            # converting data to float type and normalizing
            data = np.nan_to_num(data.astype(float))
            vis_vmin, vis_vmax = np.percentile(data, [vmin, vmax])
            norm = vis.ImageNormalize(data, vmin=vis_vmin, vmax=vis_vmax, interval=intervals[interval], stretch=stretches[stretch])

            if animation == True:
                fig, ax = plt.subplots()
                ax.set_title(search('yr\d+', file).group(), weight='bold', fontsize=17)
                ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
                plt.tight_layout(pad=0, h_pad=0, w_pad=2)
                plt.show()

                time.sleep(0.1)
                if file != files[-1]:
                    clear_output(wait=True)
            else:
                ax = axs[i // 3, i % 3]
                ax.set_title(search('yr\d+', file).group(), weight='bold', fontsize=17)
                ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
        
        if animation == False:
            plt.tight_layout(pad=0, h_pad=0.2, w_pad=0.2)
            plt.show()