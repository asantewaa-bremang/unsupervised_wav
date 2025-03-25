#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from copy import deepcopy
from scipy.signal import lfilter

import numpy as np
from tqdm import tqdm
import soundfile as sf
import os.path as osp
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
def get_parser():
    parser = argparse.ArgumentParser(description="compute vad segments")
    parser.add_argument(
        "--rvad-home",
        "-r",
        help="path to rvad home (see https://github.com/zhenghuatan/rVADfast)",
        required=True,
    )

    return parser


def rvad(speechproc, path):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    data, fs = sf.read(path)
    #logging.debug(f" shape{data.shape}")
    if len(data.shape) == 2:
          #logging.debug('it is greater')
          data = np.mean(data, axis=1)


    assert fs == 16_000, "sample rate must be 16khz"
    # ft, flen, fsh10, nfr10 = speechproc.sflux(data, int(fs* winlen), int(fs*ovrlen), nftt)
    ft,n_frames = speechproc.sflux(data, int(fs* winlen), int(fs*ovrlen), nftt)
    # logging.info("speechprocs",speechproc.sflux.__doc__)  # Prints the function's documentation

    # result = speechproc.sflux(data, int(fs * winlen), int(fs * ovrlen), nftt)
    # logging.info("printing something........",type(result), len(result))  # Check how many values are returned

    #........changing audio to monologue was not part of the code 
    #.............will delete later....................
    #logging.debug(ft)

    # --spectral flatness --
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)

    #pvblk = speechproc.pitchblockdetect(pv01, pitch, nftt, opts)
    pvblk = speechproc.pitch_block_detect(pitch, n_frames)
    #logging.debug(pvblk)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    # --pass 1--
    #signal, n_frames, frame_length, frame_shift, energy_floor, pitch_voiced
    '''noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
        fdata, n_frames, int(fs*winlen), int(fs*ovrlen), ENERGYFLOOR,pvblk
    )

    # sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0'''
    #..... old code above ........

    fdata = speechproc.snre_highenergy(
        data, n_frames, int(fs*winlen), int(fs*ovrlen), ENERGYFLOOR,pvblk
    ) 
    #logging.debug(fdata)
   

    vad_seg = speechproc.snre_vad(
       data, n_frames, int(fs*winlen), int(fs*ovrlen), ENERGYFLOOR, fdata, vadThres
    ) # not sure to use data or fdata as input
    return vad_seg, data


def main():
    parser = get_parser()
    args = parser.parse_args()

    sys.path.append(args.rvad_home)
    import speechproc

    stride = 160
    lines = sys.stdin.readlines()
    root = lines[0].rstrip()
    for fpath in tqdm(lines[1:]):
        path = osp.join(root, fpath.split()[0])
        vads, wav = rvad(speechproc, path)
        #logging.debug(f"{vads}:{wav}")

        start = None
        vad_segs = []
        for i, v in enumerate(vads):
            #logging.debug(v)
            if start is None and v == 1:
                start = i * stride
            elif start is not None and v == 0:
                vad_segs.append((start, i * stride))
                start = None
        if start is not None:
            vad_segs.append((start, len(wav)))
        logging.debug(vad_segs)

        print(" ".join(f"{v[0]}:{v[1]}" for v in vad_segs))
        logging.debug('output before')
        logging.debug(" ".join(f"{v[0]}:{v[1]}" for v in vad_segs))


if __name__ == "__main__":
    main()
