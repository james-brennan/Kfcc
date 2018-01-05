"""
Maximum likelihood KL filter
with fcc to fix Q
"""
import numpy as np
import scipy.optimize
from kernels import *



class Kalman(object):
    def __init__(self, doy, qa, refl, kerns):
        self.doy = doy
        self.qa = qa
        self.refl = refl
        self.kerns = kerns
        self.band_uncs = np.sqrt( np.array([0.005, 0.014, 0.008, 0.005,
                                0.012, 0.006, 0.003]))
        self.F = np.matrix(np.eye(3))
        self.H = np.vstack((kerns.Isotropic, kerns.Li, kerns.Ross)).T
        self.Ro = np.diag(self.band_uncs)

        self.Timesteps = np.arange(0, doy.shape[0])

        """
        get some initial estimate
        """
        # forwards
        """
        solve each band
        """
        mindoy = self.doy.min()
        idx = np.where(self.doy[qa]<mindoy+25)[0]
        cov = []
        x0 = []
        for band in xrange(7):
            # solve forwards
            A = self.H[self.qa][idx]
            y = self.refl[self.qa, band][idx]
            x0_band = np.linalg.lstsq(A, y)[0]
            #x0_band = scipy.optimize.lsq_linear(A, y, bounds=((0, 0,0), (np.inf, np.inf, np.inf))).x
            cov_band = np.linalg.inv((1/self.band_uncs[band])*A.T.dot(A))
            cov.append(cov_band)
            x0.append(x0_band)
        x0 = np.array(x0).flatten()
        Cov0 = scipy.linalg.block_diag(cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6])

        self.x0 = x0
        self.Cov0 = Cov0
        self.S = np.eye(7)
        self.invS = np.eye(7)
        self.Hk = np.zeros((21, 7))
        for b in xrange(7):
            self.Hk[3*b:(3*b+3), b]=1


    def fccModel(self, pre_fire, post_fire, pre_unc, post_unc, unc=False):
        """
        Solve fcc model on the day to day difference
        to solve for solution. fcc model is:
            rho_post - rho_pre = fcc(a_0 + a_1 f(ll) - rho_post)
        where f(ll) is a function of wavelength ll to represent
        the soil line:
            f(ll) =  (2.0 / llmax) * (ll - ll*ll/(2.0*llmax))
        to solve fcc as a linear system we can write the system as
            rho_post - rho_pre  = fcc*a_0 + fcc*a_1*f(ll) - fcc*rho_pre
        """
        wavelengths = np.array([645., 858.5, 469., 555., \
                                    1240., 1640., 2130.])
        """
        sort out uncertainty
        """
        chngUnc = 2*np.sqrt(pre_unc**2 + post_unc**2) ## ((bu*pre_unc)**2 + (bu*post_unc)**2)
        """
        calculate the values of f(ll)
        """
        loff = 400.
        lmax = 2000.
        ll =  wavelengths - loff
        llmax = lmax-loff
        lk = (2.0 / llmax) * (ll - ll*ll/(2.0*llmax))
        K = np.array(np.ones([7,3]))
        K[:, 1] = lk.transpose()#/np.sqrt(chngUnc)
        K[:, 0] = K[:, 0]#/np.sqrt(chngUnc)
        # change signal
        y = np.array(post_fire - pre_fire)#/np.sqrt(chngUnc)
                                            #Difference Post
                                            # and pre fire rhos
        """
        also need to treat change signal by including
        date uncertainty
        """
        # add third term
        K[:, 2] = pre_fire.squeeze()#/np.sqrt (chngUnc ) # K is the matrix with our linear
                                    # system of equations (K*x = y)

        """
        Make covariance matrix
        """
        C = np.diag(1/chngUnc**2)

        #import pdb; pdb.set_trace()
        KTK = K.T.dot(C).dot(K)
        KTy = K.T.dot(C).dot(y)
        sP = np.linalg.solve(KTK, KTy)
        #sP,residual, rank,singular_vals = np.linalg.lstsq ( K, y )
        #Uncertainty
        if unc:
            (fccUnc, a0Unc, a1Unc ) = \
                np.linalg.inv ( KTK ).diagonal().squeeze()
        else:
            (fccUnc, a0Unc, a1Unc ) = -998, -998,-998
        # get indiv. elements
        fcc = -sP[2]
        a0 = sP[0]/fcc
        a1 = sP[1]/fcc
        sBurn = a0 + lk*a1
        sFWD = pre_fire*(1-fcc) + fcc*sBurn
        """
        Calculate rmse
        """
        sse = np.sum( (K.dot(sP) - y)**2 )
        rmse = np.sqrt( sse/7.0)
        """
        Return log likelihood
        log l = Sum (e)**2/2*sigma**2
        """
        e = (K.dot(sP) - y)**2
        #import pdb; pdb.set_trace()
        logL = np.sum(-e / 2 * chngUnc**2)
        return fcc, a0, a1, fccUnc, a0Unc, a1Unc, rmse, logL


    def _fit(self, params, rr=False):
        """
        Estimate the process model uncertainty
        Q via optimisation
        """

        """
        Make evolution uncertainty matrix
        Q [7*3, 7*3]

        -- These are estimate by MLE
        """
        qiso = np.exp(params)*self.band_uncs #np.exp(params) # params to estimate
        qq = np.array([ [i, i*1e-1 , i*1e-1] for i in qiso]).flatten()
        Q = np.diag(qq)
        """
        set-up
        """
        # also store log at each step
        logL = []
        II = np.eye(21)
        x = []
        cov = []
        xt = self.x0.flatten()
        covT = self.Cov0
        L = []
        ys = []
        ress = []
        res = 0


        Hk = self.Hk
        invS = self.invS
        S = self.S
        """
        Run
        """
        for t in self.Timesteps:
            """
            predict
            """
            if qa[t]:
                """
                use fcc to improve convergence
                """
                cov0 = np.copy(covT)
                x0 = np.copy(xt)
                xt =  x0
                covT = cov0 + Q
                # Residuals
                for b in xrange(7):
                    Hk[3*b:(3*b+3), b]=self.H[t]
                # supposedly faster
                #np.place(Hk, Hk>0, H[t])
                y = self.refl[t] - self.Hk.T.dot(xt)
                ys.append(y)
                # Innovation covariance
                S = self.Ro + Hk.T.dot(covT).dot(Hk)
                """
                Add the residual check from
                the DA tutorial one
                """
                np.fill_diagonal(invS, 1/np.diagonal(S))
                #invS = np.linalg.inv(S)
                K = covT.dot(Hk).dot(invS)
                # Updated posterior estimate
                xt = xt + K.dot(y)
                # Updated posterior covariance
                covT = (II - K.dot(Hk.T)).dot(covT)
                # better formula
                #covT = covT - K.dot(Hk).dot(covT) - covT.dot(Hk.T).dot(K.T) + K.dot(S).dot(K.T)
                """
                And compute the new residual
                """
                e = self.refl[t] - Hk.T.dot(xt)
                """
                supposedly faster inv
                -- doesn't work
                """
                #inv_M = np.linalg.inv(covT)
                choleskey , _ = scipy.linalg.lapack.dpotrf(covT, False, False)
                inv_M , info = scipy.linalg.lapack.dpotri(choleskey)
                if info !=0:
                    inv_M = np.linalg.inv(covT)
                """
                try and reduce calculations
                by exploiting block diag structure
                """
                #H_ = Hk[:3, 0]
                #bx = [H_.T.dot(inv_M[3*b:(3*b+3), 3*b:(3*b+3)]).dot(H_) for b in xrange(7)]
                #bdet = [det3(covT[3*b:(3*b+3), 3*b:(3*b+3)]) for b in xrange(7)]
                #logT = np.log(np.prod(bdet))+ (e**2*bx).sum()
                """
                Fast log det from choleskey of the cov matrix
                """
                bdet = np.sum(2*np.log(np.diagonal(choleskey)));
                logT =  bdet + y.T.dot(Hk.T.dot(inv_M).dot(Hk)).dot(y)
                #logT = np.log(np.linalg.det(covT)) + y.T.dot(Hk.T.dot(inv_M).dot(Hk)).dot(y)
                logL.append(logT)
            x.append(np.array(xt).flatten())
            cov.append(covT)
        else:
            """
            No observations
            So just use normal model
            """
            ress.append(np.zeros(7))
            xt =  xt
            covT = covT + Q
            ys.append(0)
            x.append(np.array(xt))
            cov.append(covT)
        """
        put together
        """
        x = np.array(x)
        cov = np.array(cov)
        ress = np.array(ress)
        logL = np.array(logL)
        m = logL > 0
        Likelihood = -np.nansum(np.array(logL).flatten())
        if rr:
            return Likelihood, x, logL
        else:
            return Likelihood#, x, logL


    def _solve(self, p):
        """
        fit the model
        """
        """
        Make evolution uncertainty matrix
        Q [7*3, 7*3]

        -- These are estimate by MLE
        """
        qiso = np.exp(p)*self.band_uncs #np.exp(params) # params to estimate
        qq = np.array([ [i, i*1e-3 , i*1e-3] for i in qiso]).flatten()
        Q = np.diag(qq)

        Hk = self.Hk

        Hk = self.Hk
        invS = self.invS
        S = self.S
        #Hk = scipy.sparse.csc_matrix(np.zeros((21, 7)))
        """
        set-up
        """
        # also store log at each step
        logL = []
        II = np.eye(21)
        x = []
        cov = []
        xt = self.x0.flatten()
        covT = self.Cov0
        L = []
        ys = []
        res = 0
        yhat = []
        for b in xrange(7):
            Hk[3*b:(3*b+3), b]=1

        SSE = []
        """
        Run
        """
        qn = []
        xtol = 1e-4
        fc = []

        fcc_t = []
        for t in self.Timesteps:
            """
            predict
            """
            previous  = 0, 0, 0, 0, 0, 0, 0, 0
            if qa[t]:
                """
                use fcc to improve convergence
                """
                Qn = np.copy(Q)
                sse_old = 1e10
                sse_new = 1e9
                cov0 = np.copy(covT)
                x0 = np.copy(xt)
                xt =  x0
                fcc_iter = 0
                previous  = 0, 0, 0, 0, 0, 0, 0, 0
                while (sse_old - sse_new > xtol):# or (fcc_iter<30):
                    sse_old = sse_new
                    covT = cov0 + Qn
                    # Residuals
                    for b in xrange(7):
                        Hk[3*b:(3*b+3), b]=self.H[t]
                    # supposedly faster
                    #np.place(Hk, Hk>0, H[t])
                    y = self.refl[t] - Hk.T.dot(xt)
                    ys.append(y)
                    # Innovation covariance
                    S = self.Ro + Hk.T.dot(covT).dot(Hk)
                    # Kalman gain
                    #invs = 1/np.diag(S)
                    """
                    Add the residual check from
                    the DA tutorial one
                    """
                    np.fill_diagonal(invS, 1/np.diagonal(S))
                    #invS = np.linalg.inv(S)
                    # observation ok...
                    # use it
                    #K = covT.dot(Hk) * invs
                    K = covT.dot(Hk).dot(invS)
                    # Updated posterior estimate

                    xt__ = xt + K.dot(y)
                    # Updated posterior covariance
                    covT__ = (II - K.dot(Hk.T)).dot(covT)
                    # better formula
                    #covT = covT - K.dot(Hk).dot(covT) - covT.dot(Hk.T).dot(K.T) + K.dot(S).dot(K.T)
                    """
                    And compute the new residual
                    """
                    e = self.refl[t] - Hk.T.dot(xt__)
                    sse_ = np.sum(e**2)
                    sse_new = sse_
                    """
                    now compute fcc
                    """
                    fcc, a0, a1, fccUnc, a0Unc, a1Unc, ssef, logL = self.fccModel(x0[::3], xt__[::3], np.diag(cov0)[::3], np.diag(covT__)[::3])
                    #sse_new = ssef
                    """
                    if fcc > 0 and a1 and a0 sensible
                    increase Q
                    """
                    if (np.logical_and(fcc>-0.0, fcc<1.1)) and (np.logical_and(a0>0, a0<0.3)) and (np.logical_and(a1>0, a1<0.4)):
                        """
                        increase Q for this round by a factor
                        """
                        Qn *= (1+1e-7)
                        #xprev = xt
                        #covprev = covT
                        fcc_iter += 1
                        #import pdb; pdb.set_trace()
                        previous  = fcc, a0, a1, fccUnc, a0Unc, a1Unc, ssef, logL
                    else:
                        """
                        break fcc is getting weird
                        solve again to get the uncertainties
                        """
                        #import pdb; pdb.set_trace()
                        #if fcc_iter>0:
                        #    xt  = xprev
                        #    covT = covprev
                        fcc, a0, a1, fccUnc, a0Unc, a1Unc, ssef, logL = previous
                        break
                    # confirm update to the state
                    xt = xt__
                    covT = covT__
                """
                done iteration

                -- Finally check that observation
                is not some weird outlier
                """
                #print t, fcc_iter, "cond FALSE", fcc, a0,a1, sse_old - sse_new
                y = self.refl[t] - Hk.T.dot(xt)
                residual_sd = np.sqrt((y.T * invS* y)/np.sqrt(2*np.pi*S))[0,0]
                nirr = y[1]**2 * invS[1,1]
                res = np.sqrt(nirr)/np.sqrt(2*np.pi*S[1,1])
                #ress.append(res)
                res=np.sqrt((y**2).dot(invS))/ np.sqrt(2*np.pi*np.diag(S))
                yhat.append(Hk.T.dot(xt))
                if np.any(res>3):
                    """
                    Probably a cloud?
                    """
                    xt =  x0
                    covT = cov0 + Q
                    ys.append(0)
                    x.append(np.array(xt))
                    cov.append(covT)
                    #of  = previous
                    """
                    Add fcc stuff
                    """
                    #of = fccModel(x0[::3], xt[::3], np.diag(cov0)[::3], np.diag(covT)[::3])
                    of = previous
                    fc.append([0, 0, 0, 0, 0, 0, 0, 0])
                else:
                    """
                    obs are ok
                    """
                    x.append(np.array(xt).flatten())
                    cov.append(covT-Qn)
                    qn.append(Qn[0,0])
                    """
                    Add fcc stuff
                    """
                    #of = fccModel(x0[::3], xt[::3], np.diag(cov0)[::3], np.diag(covT)[::3])
                    if fcc_iter==1:
                        fc.append([0, 0, 0, 0, 0, 0, 0, 0])
                    else:
                        of = previous
                        fc.append(of)
            else:
                """
                No observations
                So just use normal model
                """
                xt =  xt
                covT = covT + Q
                x.append(np.array(xt))
                cov.append(covT)
                fc.append([0, 0, 0, 0, 0, 0, 0, 0])
        x = np.array(x)
        cov = np.array(cov)
        fc = np.array(fc)
        self.qn = np.array(qn)
        self.yhat = np.array(yhat)
        return x, cov, fc



#from utils import *

#doy, refl, qa, kerns = loader("mod09_h08v05_x461_y903_MTBS_fire.npz")
if __name__ == "__main__":

    def loader2(f = "pix.A2004153.r43.c34.d165.txt"):
        """
        load a fire from
        /data/geospatial_19/ucfajlg/fire/Angola/time_series
        """
        a = np.genfromtxt(f, names=True)
        refl = np.array([a['b%02i' %bb ] for bb in xrange(1, 8)]).T
        doy = a['BRDF']
        sza = a['SZA']
        vza = a['VZA']
        raa = a['RAA']
        qa = a['QA_PASS'].astype(bool)
        """
        make unique -- one a day
        """
        _, idx = np.unique(doy, return_index=True)

        qa = qa[idx]
        refl = refl[idx]
        vza = vza[idx]
        sza = sza[idx]
        raa = raa[idx]
        doy = doy[idx]
        kerns = Kernels(vza, sza, raa,
            LiType='Sparse', doIntegrals=False,
            normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
            RossType='Thick',nbar=0.0)
        return doy, qa, refl, kerns

    import glob
    ff = glob.glob("/data/geospatial_19/ucfajlg/fire/Angola/time_series/pix.*txt")

    #works: 223 221 228 1001
    # nah: 2391 2388
    fil = ff[650]
    dob = int(fil.split(".")[-2][1:])
    doy, qa, refl, kerns  = loader2(fil)

    #a = np.load("fire.npz")
    #ocals().update(a)
    #refl = refl * 0.0001

    k = Kalman(doy, qa, refl, kerns)

    p0 = np.log(1e-3*np.ones(7))
    #l, x, ll = k._fit(p0)
    #res = scipy.optimize.minimize(k._fit, p0, tol=1e-12, method='L-BFGS-B', options={'disp':True})
    #l, x, ll = k._fit(res.x, rr=True)
    x, c, f = k._solve(p0)

    plt.figure()
    plt.plot(doy[qa], refl[qa, b], 'b+-',)
    plt.plot(doy[qa], k.yhat[:, b], 'r+-',)





    plt.figure()
    plt.plot(doy, f[:, 0])
    b = 1
    plt.plot(doy[qa], refl[qa, b], 'b+-', alpha=0.4)
    i =  x[::, b*3]
    u  = c[:, b*3, b*3]
    plt.plot(doy,i, 'k-o')
    plt.plot(doy, i+u, 'k--')
    plt.plot(doy, i-u, 'k--')
    plt.axvline(dob, 0,1, color='b')
