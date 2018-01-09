# -*- coding: utf-8 -*-
# This file is part of XXX.
# https://github.com/XXX
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2017, James Brennan <james.brennan.11@ucl.ac.uk>

"""
Tikhonov.py

implementation of tikhonov regularisation
"""
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import matplotlib.pyplot as plt
import datetime
import scipy.optimize
import cProfile as profile
import sys
reload(sys)
sys.setdefaultencoding('UTF-8') 


class regularisation(object):
    """
    Performs linear least squares inversion with the kernels
    according to
    min ||Gm-d||^{2}_{2} + Î»||L||^{2}_{2}
    which is:
    where
    G is the kernels matrix
    d is the observations
    L is the regularisation matrix (a first difference matrix)
    Î» is the regularisation parameter which is estimated with the gcv
    """
    def __init__(self, date, rho, qa, sza, vza, raa, regMatrix=None):
        """
        """
        # unc from http://modis-sr.ltdri.org/pages/validation.html
        self.band_uncs = np.array([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])
        self.doy = date
        self.rho = rho
        self.qa = qa
        #self.qa = np.logical_and(qa, np.all(self.rho>0, axis=1)) #qa
        """
        get only one osbervation per day
        -- I don't yet know how to do both
            --> but probably worth it
        for now just select whichever is true each day
        """
        _i = 0
        idxs = []
        _date = date[0]
        while _date < date[-1]:
            idx = (date - _date)==datetime.timedelta(0) 

            qidx = np.where(self.qa *idx)[0]
            if qidx.size==0:
                # no obs
                idxs.append(_i)
                _i +=1
            else:
                #import pdb; pdb.set_trace()
                idxs.append(qidx[0] )
                _i = qidx[0]
            _date += datetime.timedelta(1) 
        idxs.append(_i+1)
        idxs = np.array(idxs)
        #self.uniqDates, uniqIndx = np.unique(self.doy, return_index=True)
        self.rho = self.rho[idxs]
        self.qa = self.qa[idxs]
        self.DOY = date[idxs]
        self.vza = vza[idxs]
        self.sza = sza[idxs]
        self.raa = raa[idxs]
        kerns =  Kernels( self.vza, self.sza, self.raa, \
                        LiType='Sparse', doIntegrals=False, \
                        normalise=1, RecipFlag=True, RossHS=True, MODISSPARSE=True, \
                        RossType='Thick' )
        self.dIndex = np.array(xrange(len(self.DOY)))
        """
        Fix qa to exclude some negative observations getting through
        """
        self.nT = self.qa.shape[0]
        self.I = np.eye(3*self.nT)
        self.L = None
        if regMatrix == None:
            # make the regularisation matrix
            self.L = self.regMatrix()
            self.L = self.L.T.dot(self.L)
            self.L = self.L.T.dot(self.L)
            self.LT = self.L.T
        """
        Pre-make the kernels
        """
        # Calculate the relevant kernels.
        self.K_obs =  copy.deepcopy(kerns)
        self.kold =  copy.deepcopy(kerns)

        """
        make kernels correct equal weighting
        """
        self.isonorm = np.linalg.norm( self.K_obs.Isotropic[self.qa])
        self.rossnorm = np.linalg.norm( self.K_obs.Ross[self.qa])
        self.linorm = np.linalg.norm( self.K_obs.Li[self.qa])
        self.K_obs.Isotropic /= self.isonorm
        self.K_obs.Ross /=  self.rossnorm
        self.K_obs.Li /= self.linorm


        self._xsols = np.zeros((7, 3*self.nT))

    def regMatrix(self):
        """
        *-----------------------*
         Make regulariser matrix
        *-----------------------*
        """
        nx=self.nT # 365*3+90 for the wings
        lag = 1
        I = np.eye(nx) #np.diag (np.ones(nx))
        D = (I - np.roll ( I, -lag)).T
        D2 = D.T.dot(D)
        # try change boundary
        D2[self.nT-1, self.nT-1]=2
        D2[self.nT-2, self.nT-1]=-2
        Z = np.zeros_like (I)
        DD0 = np.array([D2, Z, Z]).reshape ( nx*3, nx).T
        DD1 = np.array([Z, D2, Z]).reshape ( nx*3, nx).T
        DD2 = np.array([Z, Z, D2]).reshape ( nx*3, nx).T
        DD = np.array( [ DD0, DD1, DD2]).reshape ((nx*3, nx*3))
        DD = sp.csc_matrix( DD )
        return DD

    def kernelMatrix(self, band=0):
        """
        Construct the kernels matrix
        :param band: band chosen
        :returns: :math:`\mathbf{K}`
            (and also the covariance matrix)
        """
        K_obs = self.K_obs
        n_kernels = 3
        n_bands = 1
        K = sp.lil_matrix((self.nObs, n_kernels*self.nT))
        cov_mat = np.zeros((self.nObs, 1))
        """
        Figure out where to put the kernel elements in the matrix
        """
        whereToPut = np.where(self.qa==True)
        whereToPut=zip(*whereToPut)
        i_obs = -1
        # return the uncertainty components
        self.atm_unc = []
        self.cov_unc = []
        unc = self.band_uncs[band]
        for ob in self.rho[self.qa, band]: 
            i_obs += 1
            """
            place kernels at correct locations
            """
            wh = whereToPut[i_obs]
            nti = wh[0]
            ix = i_obs
            K [ i_obs, (nti)]  = K_obs.Isotropic[ix] # Isotropic
            K [ i_obs, self.nT+(nti)] = \
                                K_obs.Ross[ix]
            K [ i_obs, 2*self.nT+nti ] = \
                                K_obs.Li[ix]
            """
            Fill the observation uncertainty matrix C
            according to MOD09c6 atmospheric correction uncertainty
            is ï¿½(0.005 + 0.05*reflectance)
            Also lets add more uncertainty to observations 
            with low observation coverage (%pixel observation covers)
            -- not sure what to use but this seems justifiable
            o_sigma = (1-0.68) * rho * (1 - 0.90)**2
            this re-scales rho sigma (2sqd) based on the value 
            of obscov. When obscov is 1, o_sigma is 0. When it 
            is 0 
            """
            cov_mat[i_obs, :] = (1./unc ) 
        covariance_matrix = sp.lil_matrix((self.qa.sum(),
                    self.qa.sum()))
        covariance_matrix.setdiag(cov_mat)
        return K.tocsc(), covariance_matrix.tocsc()

    def dVec(self, band=0,):
        """
        make vector of weighted observations
        also return observations au natural
        """
        d = sp.lil_matrix((self.nObs, 1))
        i_obs = 0
        for ob in self.rho[self.qa, band]:
            d[i_obs, :] = ob
            i_obs += 1
        y = self.rho[self.qa, band]

        #d = np.zeros(self.nT)
        #d[self.qa] = self.rho[self.qa, band]

        return d, y

    def _prepare_matrices(self, band=0):
        """
        Builds matrices
        """
        # we need a few things
        if self.L == None :
            # no point re-making matrices which don't change
            self.L = self.regMatrix()
            self.LT = self.L.T
        self.nObs = self.qa.sum()
        self.G, self.cov_mat = self.kernelMatrix(band=band)
        self.d, self.y = self.dVec(band=band)
        # also do transforms
        self.GT = self.G.T
        return None

    def _solve(self, alpha=1e6):
        """
        function for solving the system
        """
        reg = alpha*self.L
        self.reg = reg
        # make a matrix
        self.A = (self.GT*self.cov_mat*self.G + self.reg).tocsc()
        self.AI = sp.linalg.splu ( self.A )
        self.b = (self.GT*self.cov_mat*self.d).toarray().squeeze()
        xsol = self.AI.solve ( self.b)
        return xsol


    def getUncertainties(self):
        """
        Sometimes SuperLU might fail to invert the
        A matrix for the hessian.
        As a backup use numpy
        """
        self.Ainv = np.linalg.inv(self.A.todense())
        diag = self.Ainv.diagonal()
        diag = np.array(diag).reshape(-1)
        iso_unc = diag[:self.nT]
        geo_unc = diag[self.nT:2*self.nT]
        vol_unc = diag[2*self.nT:3*self.nT]
        return {'iso': iso_unc,
                'geo': geo_unc,
                'vol': vol_unc}

    def solve(self, band=1, alpha=1e1,
    		calculate_uncs=False):
        """
        Main fitting Function
        """
        self.xsols = {'iso':{}, 'geo':{}, 'vol':{}}
        self.uncs = {'iso':{}, 'geo':{}, 'vol':{}}
        """
        Estimate lambda using just nir band (band 2)
        """
        self._prepare_matrices(band=band)
        xsol = self._solve(alpha=alpha)
        # pull out individual kernels
        iso = xsol[:self.nT]
        vol = xsol[self.nT:2*self.nT]
        geo = xsol[2*self.nT:3*self.nT]
        self.iso = iso / self.isonorm
        self.geo = geo / self.linorm
        self.vol = vol / self.rossnorm
        self.x = np.vstack((self.iso, self.vol, self.geo)).T
        # same for uncertainties
        if calculate_uncs:
            uncs = self.getUncertainties()
            self.uncs['iso'][band]=uncs['iso']
            self.uncs['geo'][band]=uncs['geo']
            self.uncs['vol'][band]=uncs['vol']

    def fit(self, qa, **kwargs):
        """
        """
        band = 1
        self.qa = qa
        alpha = kwargs['alpha']
        # solve
        self.solve(band=1, alpha = alpha)
    
    def predict(self, qa, band=1):
        """
        this is used for cv
                
        Arguments:
            qa {[type]} -- [description]
        """
        
        self.qa = qa
        """
        predict observations
        """
        K = np.vstack((self.kold.Isotropic[self.qa],
                    self.kold.Ross[self.qa],
                    self.kold.Li[self.qa], ))
        #import pdb; pdb.set_trace()
        pred = (K.T*self.x[self.qa]).sum(axis=1)
        return pred

    def chooseAlpha(self, band=1):
        """
        Use the method proposed in the original
        Quaife Lewis paper
        """
        band_uncs = np.array([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])
        alphas = np.logspace(1, 4, 20)
        obs = self.rho[self.qa, band]
        n = self.qa.sum()
        std  = []
        for al in alphas:
            self.solve(band=1, alpha = al)
            pred = self.predict(band=1, qa=self.qa)
            res = pred - obs
            std.append(res.std())
        self.std = np.array(std)
        chosen = alphas[np.argmin(np.abs(self.std-band_uncs[band]))]
        return chosen

  