from nilearn import image, plotting
from matplotlib import colormaps, rcParams
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import os
import os.path as op
import shutil
import subprocess as sp
import logging
import bs4

log = logging.getLogger(__name__)

def get_spectrum(data: np.array, tr: float = 1.0):
    """
    Return the power spectrum and corresponding frequencies.

    Done when provided with a component time course and repitition time.

    Parameters
    ----------
    data : (S, ) array_like
            A timeseries S, on which you would like to perform an fft.
    tr : :obj:`float`
            Reptition time (TR) of the data
    """
    # adapted from @dangom
    power_spectrum = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(power_spectrum.size * 2 - 1, tr)
    idx = np.argsort(freqs)
    return power_spectrum[idx], freqs[idx]


def component_images(analysis_dir):
    """
    Creates static figure of component classification and features. Used for visual inspection of the results.
    Inputs:
        analysis_dir - Pathlike or sting
    """
    # all filename are consistent across runs
    melodic_filename = op.join(analysis_dir, 'melodic_IC_thr_MNI2mm.nii.gz')
    mmix_filename = op.join(analysis_dir, 'melodic.ica', 'melodic_mix')
    meanfunc_filename = op.join(analysis_dir, 'melodic.ica', 'mean.nii.gz')
    stats_filename = op.join(analysis_dir, 'melodic.ica', 'melodic_ICstats')
    labels_filename = op.join(analysis_dir, 'classification_overview.txt')

    if not op.exists(melodic_filename) or not op.exists(labels_filename):
        log.error("unable to locate ICA-AROMA results file.")
        return

    if not op.exists(mmix_filename) or not op.exists(meanfunc_filename) or not op.exists(stats_filename):
        log.error("unable to locate ICA-AROMA working directory.")
        return

    # read in output files
    mmix = pd.read_csv(mmix_filename, delim_whitespace=True, header=None)
    mean_func = nib.load(meanfunc_filename)
    tr = mean_func.header["pixdim"][4]
    time = np.linspace(0, mmix.shape[0] * tr, mmix.shape[0])

    stats = pd.read_csv(stats_filename, delim_whitespace=True, header=None)
    labels = pd.read_csv(labels_filename, sep='\t')

    # loop through all components in 4D melodicIC file and create plot
    for idx, img in enumerate(image.iter_img(melodic_filename)):

        # compute values for plotting...
        imgmax = 0.75 * np.abs(img.get_fdata()).max()
        imgmin = imgmax * -1

        allplot = plt.figure(figsize=(12, 3))
        plt.subplots_adjust(hspace=0.8)
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1, fig=allplot)

        # add label
        # CXX [noise]: Tot. var. expl XX%
        comp_var = f"{stats[0][idx]:.2f}"
        comp_label = labels["Motion/noise"][idx]
        compnum = labels["IC"][idx]
        if comp_label == True:
            plt_title = (
                f"Comp. {compnum} [noise]: variance: {comp_var}%"
            )
            plotcolor = 'red'
        elif comp_label == False:
            plt_title = (
                f"Comp. {compnum}: variance: {comp_var}%"
            )
            plotcolor = 'green'

        rcParams['text.color'] = plotcolor
        rcParams['axes.titlecolor'] = plotcolor
        rcParams['axes.labelcolor'] = plotcolor
        rcParams['xtick.color'] = plotcolor
        rcParams['ytick.color'] = plotcolor

        title = ax1.set_title(plt_title, loc='left')
        title.set_y(0.8)

        # glass brain
        plotting.plot_glass_brain(img, threshold=0.1 * imgmax, cmap=colormaps['jet'], vmax=imgmax, vmin=imgmin,
                                  plot_abs=False, axes=ax1)

        # time series
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1, fig=allplot)
        ax2.plot(time, mmix[idx], linewidth=0.5, color=plotcolor)

        ax2.set_xlabel("seconds")
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        plt.tick_params(left=False, bottom=False, labelleft=False)

        # fft
        ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1, fig=allplot)
        spectrum, freqs = get_spectrum(mmix[idx], tr)
        ax3.plot(freqs, spectrum, color=plotcolor)
        ax3.set_xlabel("Hz")
        ax3.set_xlim(freqs[0], freqs[-1])

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        plt.tick_params(left=False, bottom=False, labelleft=False)

        # save figure...
        os.makedirs(op.join(analysis_dir, "figures"), exist_ok=True)
        fname = os.path.join(analysis_dir, "figures", "C" + str(compnum).zfill(2) + ".png")
        plt.savefig(fname, format='png', bbox_inches='tight', pad_inches=0.25)

        plt.close(allplot)

    return


def carpet_plots(input_img_filename, output_img_filename, analysis_dir):
    """
    Creates static figure of fMRI image before and after denoising. Used for visual inspection of the results.
    Inputs:
        input_img_filename - functional image used for ICAAROMA
        output_img_filename - denoised output from ICAAROMA
        analysis_dir - Pathlike or sting
    """

    if not op.exists(input_img_filename):
        log.error("Input image passed for report generation does not exist.")
        return

    if not op.exists(output_img_filename):
        log.error("Input image passed for report generation does not exist.")
        return

    carpetplot = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(hspace=0.25)

    ax1 = plt.subplot2grid((2, 1), (0, 0), fig=carpetplot)
    func = image.load_img(input_img_filename)
    tr = func.header["pixdim"][4]
    plotting.plot_carpet(
        func,
        t_r=tr,
        standardize=False,
        axes=ax1,
        title="Before"
    )
    plt.axis('off');

    ax2 = plt.subplot2grid((2, 1), (1, 0), fig=carpetplot)
    func = image.load_img(output_img_filename)
    tr = func.header["pixdim"][4]
    plotting.plot_carpet(
        func,
        t_r=tr,
        standardize=False,
        axes=ax2,
        title="After"
    )

    plt.axis('off');

    # save figure...
    os.makedirs(op.join(analysis_dir, "figures"), exist_ok=True)
    fname = os.path.join(analysis_dir, "figures", "carpetplot.png")
    plt.savefig(fname, format='png')

    return


def report(gear_options, app_options):

    # generate report for ICA-AROMA
    # copy report to working directory
    report_file = op.join(app_options["analysis_dir"], app_options["acqid"] + "-report.html")
    log.info("Generating report file: %s", os.path.basename(report_file))
    shutil.copy2(op.join(op.dirname(op.realpath(__file__)), "report.html"), report_file)

    # create figures...
    component_images(app_options["analysis_dir"])

    carpet_plots(gear_options["aroma"]["params"]["in"], app_options["out_file"], app_options["analysis_dir"])

    # update list of images for report...
    cwd = os.getcwd()
    os.chdir(app_options["analysis_dir"])
    files = searchfiles("figures/C*.png")
    os.chdir(cwd)

    # load html into python
    with open(report_file) as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt, "html.parser")

    for f in files:
        new_tag = soup.new_tag("img", src=f)
        elm = soup.find(attrs={'id': 'components'})
        elm.append(new_tag)

    # replace the command used to run ica-aroma
    result = soup.find(attrs={'id': 'cmd'})
    result.string.replace_with(app_options["command"])

    with open(report_file, "w") as outf:
        outf.write(str(soup))

    return


def searchfiles(path, dryrun=False, exit_on_errors=True) -> list[str]:
    cmd = "ls -d " + path

    if not dryrun:
        terminal = sp.Popen(
            cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, stderr = terminal.communicate()
        returnCode = terminal.poll()

        files = stdout.strip("\n").split("\n")

        return files

