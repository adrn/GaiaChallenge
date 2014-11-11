# coding: utf-8

""" Potential used for Pal 5 Challenge at the Gaia Challenge 2 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.constants import G
import numpy as np

# Project
# from .core import CartesianCompositePotential
from gary.potential.cpotential import CCompositePotential
from gary.potential.cbuiltin import JaffePotential, MiyamotoNagaiPotential
from gary.units import galactic

__all__ = ['GC2Pal5Potential']

# ============================================================================
#    Pal5 Challenge Halo (for Gaia Challenge)
#
cdef class _Pal5AxisymmetricNFWPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double M, Rh, qz
    cdef public double G, GM, qz2

    def __init__(self, double M, double Rh, double qz):
        """ Units of everything should be in the system:
                kpc, Myr, radian, M_sun
        """
        self.M = M
        self.Rh = Rh
        self.qz = qz

        self.qz2 = self.qz*self.qz
        self.G = 4.49975332435e-12  # kpc, Myr, Msun
        self.GM = self.G * self.M

    cdef public inline double _value(self, double[:,::1] r, int k) nogil:
        cdef double R
        R = r[k,0]*r[k,0] + r[k,1]*r[k,1] + r[k,2]*r[k,2]/self.qz2
        return -self.GM / R * log(1. + R/self.Rh)

    cdef public inline void _gradient(self, double[:,::1] r, double[:,::1] grad, int k) nogil:
        cdef double R, dPhi_dR
        R = sqrt(r[k,0]*r[k,0] + r[k,1]*r[k,1] + r[k,2]*r[k,2]/self.qz2)
        dPhi_dR = self.GM / R / R * (log(1+R/self.Rh) - R/(R+self.Rh))
        grad[k,0] += dPhi_dR * r[k,0] / R
        grad[k,1] += dPhi_dR * r[k,1] / R
        grad[k,2] += dPhi_dR * r[k,2] / R / self.qz2

class Pal5AxisymmetricNFWPotential(CPotential, CartesianPotential):
    r"""
    Flattened, axisymmetric NFW potential Andreas used for the Pal 5 challenge.
    .. math::
        \Phi &=
    Parameters
    ----------
    TODO:
    qz : numeric
        Flattening in Z direction.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    """
    def __init__(self, M, Rh, qz, units=None):
        self.units = units
        parameters = dict(M=M, Rh=Rh, qz=qz)
        super(Pal5AxisymmetricNFWPotential, self).__init__(_Pal5AxisymmetricNFWPotential,
                                                           parameters=parameters)

# change to CartesianCompositePotential for Pure-Python
class GC2Pal5Potential(CCompositePotential):

    def __init__(self, m_disk=1E11, a=6.5, b=0.26,
                 m_spher=3.4E10, c=0.3,
                 m_halo=1.81194E12, Rh=32.26, qz=0.814,
                 units=galactic):

        # Choice of v_h sets circular velocity at Sun to 220 km/s
        self.units = units

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = JaffePotential(units=units,
                                         m=m_spher, c=c)

        kwargs["halo"] = Pal5AxisymmetricNFWPotential(units=units,
                                                      M=m_halo, Rh=Rh, qz=qz)
        super(GC2Pal5Potential,self).__init__(**kwargs)
        self.c_instance.G = G.decompose(units).value
