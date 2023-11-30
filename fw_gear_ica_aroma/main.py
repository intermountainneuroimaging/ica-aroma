"""Main module."""

import logging
import os
import os.path as op
import numpy as np
import pandas as pd
import shutil
import errorhandler
from typing import List, Tuple
import nibabel as nb
from copy import deepcopy
from nipype.utils.filemanip import fname_presuffix
from flywheel_gear_toolkit import GearToolkitContext

from utils.command_line import exec_command, build_command_list
from utils.report.report import report
from utils.zip_htmls import zip_htmls
from fw_gear_ica_aroma.common import execute_shell, searchfiles
from fw_gear_ica_aroma.metadata import find_matching_acq

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()


def prepare(
        gear_options: dict,
        app_options: dict,
) -> Tuple[List[str], List[str]]:
    """Prepare everything for the algorithm run.

    It should:
     - Install FreeSurfer license (if needed)

    Same for FW and RL instances.
    Potentially, this could be BIDS-App independent?

    Args:
        gear_options (Dict): gear options
        app_options (Dict): options for the app

    Returns:
        errors (list[str]): list of generated errors
        warnings (list[str]): list of generated warnings
    """
    # pylint: disable=unused-argument
    # for now, no errors or warnings, but leave this in place to allow future methods
    # to return an error
    errors: List[str] = []
    warnings: List[str] = []

    return errors, warnings
    # pylint: enable=unused-argument


def run(gear_options: dict, app_options: dict, gear_context: GearToolkitContext) -> int:
    """Run ICA-AROMA using generic bids-derivative inputs.

    Arguments:
        gear_options: dict with gear-specific options
        app_options: dict with options for the BIDS-App

    Returns:
        run_error: any error encountered running the app. (0: no error)
    """

    log.info("This is the beginning of the run file")

    # psudocode
    # 1. setup method for running in standard (FSL space) - check if exists?
    #   a. identify func file
    #   b. create fsl formatted motion file (check if par file exists?) - this should be the correct format?
    #   c. set output directory (tempdir?) -
    #   d. run ica-aroma
    #   e. move relevant files to final destination
    # 2. if func file is not standard space - look for reg or run it?
    #   a. same as above - also pass reg mat and warp
    # 3. reporting...
    #   a. add component plot included with aroma + glass brain visualization
    # 4. zip

    outfiles = []; run_error = 0
    for acq in app_options["runs"]:

        app_options["acqid"] = acq

        # try to run with standard space images...
        lookup_table = build_lookup(gear_options, app_options)
        func_file_standard = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz",
            lookup_table)

        func_file_native = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_space-func_desc-preproc_bold.nii.gz",
            lookup_table)

        featdir = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_space-func_desc-preproc_bold.feat",
            lookup_table)

        if os.path.exists(func_file_standard):
            app_options["func_file"] = func_file_standard
            input = "standard-space"

        elif os.path.exists(func_file_native):
            app_options["func_file"] = func_file_native
            input = "native-space"

        elif os.path.exists(featdir):
            app_options["featdir"] = featdir
            input = "feat"

        else:
            log.warning("Files needed for ICA-AROMA not found...skipping: %s", acq)
            # TODO add run option for func space - first run reg
            continue

        log.info("Running ICA-AROMA: %s", os.path.basename(app_options["func_file"]))

        if input == "standard-space" or input == "native-space":

            # locate motion file...
            app_options["motion_file"] = apply_lookup(
                "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_desc-mcf_timeseries.par",
                lookup_table)

            if not os.path.exists(app_options["motion_file"]):
                app_options["confound_file"] = apply_lookup(
                    "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_desc-confounds_timeseries.tsv",
                    lookup_table)
                create_fsllike_motion(gear_options, app_options)

            # identify if dummy scans should be included
            # orig_file = app_options["func_file"]
            # if app_options['DropNonSteadyState']:
            #     app_options['AcqDummyVolumes'] = fetch_dummy_volumes(app_options["acqid"], gear_context)
            #     gear_options["aroma"]["params"]["in"] = _remove_volumes(app_options["func_file"], app_options['AcqDummyVolumes'])
            #     gear_options["aroma"]["params"]["mc"] = _remove_timepoints(app_options["motion_file"], app_options['AcqDummyVolumes'])
            # else:
            #     gear_options["aroma"]["params"]["in"] = app_options["func_file"]
            #     gear_options["aroma"]["params"]["mc"] = app_options["motion_file"]

        if input == "feat":
            gear_options["aroma"]["params"]["feat"] = app_options["featdir"]

        if input == "native-space":
            #TODO add registration and pass affmat and warp
            pass

        # output directory name
        gear_options["aroma"]["params"]["out"] = app_options["func_file"].replace(".nii.gz", ".aroma")
        app_options["analysis_dir"] = gear_options["aroma"]["params"]["out"]
        app_options["out_file"] = op.join(gear_options["aroma"]["params"]["out"], "denoised_func_data_nonaggr.nii.gz")

        # if dimensionality passed - use it here....
        if app_options["aroma-melodic-dimensionality"]:
            gear_options["aroma"]["params"]["dim"] = str(app_options["aroma-melodic-dimensionality"])

        if app_options["denoising_strategy"]:
            gear_options["aroma"]["params"]["den"] = str(app_options["denoising_strategy"])

        # Build the command and execute
        command = deepcopy(gear_options["aroma"]["common_command"])
        command = build_command_list(command, gear_options["aroma"]["params"])
        app_options["command"] = " ".join(command)
        # exec_command(
        #     command,
        #     environ=os.environ,
        #     dry_run=gear_options["dry-run"],
        #     shell=True,
        #     cont_output=True,
        # )

        # if app_options['DropNonSteadyState']:
        #     # add volumes back to output file -- add dummy values to melodicmix (time series)
        #     melodic_mix_file = op.join(app_options["analysis_dir"], "melodic.ica", "melodic_mix")
        #     if os.path.exists(melodic_mix_file):
        #         _add_volumes_melodicmix(melodic_mix_file, app_options['AcqDummyVolumes'])
        #
        #     for i in ["denoised_func_data_nonaggr.nii.gz", "denoised_func_data_aggr.nii.gz"]:
        #         f = op.join(app_options["analysis_dir"], i)
        #         if os.path.exists(f):
        #             out = fname_presuffix(f, suffix='_cut')
        #             shutil.move(f, out)
        #             _add_volumes(orig_file, out, app_options['AcqDummyVolumes'])


        # make new denoised output file
        desc = [s for s in op.basename(app_options["func_file"]).split("_") if "desc" in s][0]
        if app_options["denoising_strategy"] == "nonaggr" or app_options["denoising_strategy"] == "both":
            f = op.join(app_options["analysis_dir"],"denoised_func_data_nonaggr.nii.gz")
            newpath = app_options["analysis_dir"].replace(".aroma", ".nii.gz").replace(desc, "desc-smoothAROMAnonaggr")
            shutil.copy(f, newpath)
            outfiles.append(newpath)

        if app_options["denoising_strategy"] == "aggr" or app_options["denoising_strategy"] == "both":
            f = op.join(app_options["analysis_dir"],"denoised_func_data_aggr.nii.gz")
            newpath = app_options["analysis_dir"].replace(".aroma", ".nii.gz").replace(desc, "desc-smoothAROMAaggr")
            shutil.copy2(f, newpath)
            outfiles.append(newpath)

        if app_options["save_intermediates"] or app_options["denoising_strategy"] == "no":
            outfiles.append(app_options["analysis_dir"])

        if error_handler.fired:
            log.critical('Failure: exiting with code 1 due to logged errors')
            run_error = 1
            return run_error

        # generate report and zip html
        # report(gear_options, app_options)
        #
        # zip_htmls(gear_options["output-dir"], gear_options["destination-id"],
        #           op.join(gear_options["aroma"]["params"]["out"]))

    # move output files to path with destination-id
    for f in outfiles:
        newpath = f.replace(str(gear_options["work-dir"]), op.join(gear_options["work-dir"],gear_options["destination-id"]))
        os.makedirs(op.dirname(newpath), exist_ok=True)
        try:
            if os.path.isfile(f):
                shutil.copy2(f, newpath)
            else:
                shutil.copytree(f, newpath, symlinks=True, dirs_exist_ok=True)
        except:
            pass

    # zip results
    cmd = "zip -q -r --symlinks " + os.path.join(gear_options["output-dir"],
                                      "aroma_" + str(gear_options["destination-id"])) + ".zip " + gear_options[
              "destination-id"]
    execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["work-dir"])

    return run_error


# -----------------------------------------------
# Support functions
# -----------------------------------------------


def create_fsllike_motion(gear_options: dict, app_options: dict):
    df = pd.read_csv(app_options["confounds_file"], sep='\t')

    parfile = op.join(op.dirname(app_options["confounds_file"]),
                      "tmp.par")  # first three columns contain the rotations for the X, Y, and Z voxel axes, in radians. The remaining three columns contain the estimated X, Y, and Z translations
    pardf = df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]].copy()

    pardf.to_csv(parfile, sep=" ", header=None, index=False)

    app_options["motion_file"] = parfile


def apply_lookup(text, lookup_table):
    if '{' in text and '}' in text:
        for lookup in lookup_table:
            text = text.replace('{' + lookup + '}', lookup_table[lookup])
    return text


def build_lookup(gear_options,app_options):
    # apply filemapper to each file pattern and store
    if os.path.isdir(os.path.join(gear_options["work-dir"], "fmriprep")):
        pipeline = "fmriprep"
    elif os.path.isdir(os.path.join(gear_options["work-dir"], "bids-hcp")):
        pipeline = "bids-hcp"
    elif len(os.walk(gear_options["work-dir"]).next()[1]) == 1:
        pipeline = os.walk(gear_options["work-dir"]).next()[1]
    else:
        log.error("Unable to interpret pipeline for analysis. Contact gear maintainer for more details.")
    app_options["pipeline"] = pipeline

    lookup_table = {"WORKDIR": str(gear_options["work-dir"]), "PIPELINE": pipeline, "SUBJECT": app_options["sid"],
                    "SESSION": app_options["sesid"], "ACQ": app_options["acqid"]}

    return lookup_table


def _remove_volumes(bold_file,n_volumes):
    if n_volumes == 0:
        return bold_file

    out = fname_presuffix(bold_file, suffix='_cut')
    nb.load(bold_file).slicer[..., n_volumes:].to_filename(out)
    return out


def _remove_timepoints(motion_file,n_volumes):
    arr = np.loadtxt(motion_file, ndmin=2)
    arr = arr[n_volumes:,...]

    filename, file_extension = os.path.splitext(motion_file)
    motion_file_new = motion_file.replace(file_extension,"_cut"+file_extension)
    np.savetxt(motion_file_new, arr, delimiter='\t')
    return motion_file_new


def _add_volumes(bold_file, bold_cut_file, n_volumes):
    """prepend n_volumes from bold_file onto bold_cut_file"""
    bold_img = nb.load(bold_file)
    bold_data = bold_img.get_fdata()
    bold_cut_img = nb.load(bold_cut_file)
    bold_cut_data = bold_cut_img.get_fdata()

    # assign everything from n_volumes foward to bold_cut_data
    bold_data[..., n_volumes:] = bold_cut_data

    out = bold_cut_file.replace("_cut","")
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)
    return out


def _add_volumes_melodicmix(melodic_mix_file, n_volumes):

    melodic_mix_arr = np.loadtxt(melodic_mix_file, ndmin=2)

    if n_volumes > 0:
        zeros = np.zeros([n_volumes, melodic_mix_arr.shape[1]])
        melodic_mix_arr = np.vstack([zeros, melodic_mix_arr])

        # save melodic_mix_arr
    np.savetxt(melodic_mix_file, melodic_mix_arr, delimiter='\t')


def fetch_dummy_volumes(taskname, context):
    # Function generates number of dummy volumes from config or mriqc stored IQMs
    if context.config["DropNonSteadyState"] is False:
        return 0

    acq, f = find_matching_acq(taskname, context)

    if "DummyVolumes" in context.config:
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        log.info("Set by user....Using %s dummy volumes", context.config['DummyVolumes'])
        return context.config['DummyVolumes']

    if f:
        IQMs = f.info["IQM"]
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        if "dummy_trs_custom" in IQMs:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs_custom"])
            return IQMs["dummy_trs_custom"]
        else:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs"])
            return IQMs["dummy_trs"]

    # if we reach this point there is a problem! return error and exit
    log.error(
        "Option to drop non-steady state volumes selected, no value passed or could be interpreted from session metadata. Quitting...")