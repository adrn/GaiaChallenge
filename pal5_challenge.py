# coding: utf-8

""" Gaia Challenge 2 -- Pal 5 Challenge """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import os
import sys

# Third-party
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np
import gary.dynamics as sd
import gary.integrate as si
import gary.io as io
import gary.potential as sp
from gary.units import galactic
from gary.util import get_pool
import triangle

from astropy.constants import G
Gee = G.decompose(galactic).value

# streams
from gaiachallenge.potential import GC2Pal5Potential
from streams.rewinder import Rewinder, RewinderSampler
import streams.coordinates as stc

# -- HACK --
# TODO: need a better way to do this
import gary.potential as gp
gp.GC2Pal5Potential = GC2Pal5Potential
# -- HACK --

global pool
pool = None

def plot_traces(sampler, p0=None, truths=None):
    figs = []
    for i in range(sampler.dim):
        fig,ax = plt.subplots(1,1,figsize=(10,6))
        for chain in sampler.chain[...,i]:
            ax.plot(chain, marker=None, drawstyle='steps', alpha=0.2, color='k')

        if p0 is not None:
            for pp in p0[:,i]:
                ax.axhline(pp, alpha=0.2, color='r')

        if truths is not None:
            ax.axhline(truths[i], alpha=0.7, color='g')

        figs.append(fig)

    return figs

def main(mpi=False):
    pool = get_pool(mpi=mpi)

    cfg_path = "config/pal5_challenge.yml"
    out_path = "output/pal5_challenge"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model = Rewinder.from_config(cfg_path)
    sampler = RewinderSampler(model, nwalkers=64, pool=pool)
    model.stars.parameters['tail'] = -model.stars.parameters['tail']

    # true_parameters = dict(potential=dict(m_halo=1.81194E12, Rh=32.26, qz=0.814))
    # parameter_sigmas = dict(potential=dict(m_halo=1E11, Rh=1., qz=0.1))
    true_parameters = dict(potential=dict(m_halo=1.81194E12, Rh=32.26, qz=0.814, m_disk=1E11,
                                          a=6.5, b=0.26, m_spher=3.4E10, c=0.7),
                           hyper=dict(alpha=1.125, theta=-0.3))

    truth = model.vectorize(true_parameters)
    # p0_sigma = model.vectorize(parameter_sigmas)
    p0_sigma = np.abs(truth*1E-6)
    p0 = np.random.normal(truth, p0_sigma, size=(sampler.nwalkers, sampler.dim))

    # burn in
    sampler.run_inference(p0, 250)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    sampler.reset()
    logger.info("Done burning in")

    # restart walkers from best position, burn again
    new_pos = np.random.normal(best_pos, p0_sigma,
                               size=(sampler.nwalkers, p0.shape[1]))
    sampler.run_inference(new_pos, 500)
    pos = sampler.chain[:,-1]
    sampler.reset()

    # run for inference steps
    sampler.run_inference(pos, 500)

    figs = plot_traces(sampler, p0=None, truths=truth)
    for i,fig in enumerate(figs):
        fig.savefig(os.path.join(out_path, "{}.png".format(i)))

    logger.debug("Acceptance fraction: {}".format(sampler.acceptance_fraction))

    fig = triangle.corner(sampler.flatchain[:,:3], truths=truth[:3],
                          extents=[(1.E12,3E12),(20,40),(0.75,0.86)],
                          labels=[r"$M$ [$M_\odot$]", r"$R_h$ [kpc]", "$q_z$"])
    fig.savefig(os.path.join(out_path, "corner.png"))

    pool.close()
    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        main(mpi=args.mpi)
    except:
        pool.close() if hasattr(pool, 'close') else None
        raise
        sys.exit(1)

    pool.close() if hasattr(pool, 'close') else None
    sys.exit(0)
