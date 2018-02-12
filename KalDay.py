import numpy as np

def fccModel(pre_fire, post_fire, pre_unc, post_unc, unc=False):
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
    # turn into standard devs
    #pre_unc = np.sqrt(pre_unc)
    #post_unc = np.sqrt(post_unc)
    #chngUnc = 2*np.sqrt(pre_unc**2 + post_unc**2) ## ((bu*pre_unc)**2 + (bu*post_unc)**2)
    chngUnc = np.sqrt(pre_unc**2 + post_unc**2)
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
    CInv = np.diag(1/chngUnc**2)

    #import pdb; pdb.set_trace()
    KTK = K.T.dot(CInv).dot(K)
    KTy = K.T.dot(CInv).dot(y)
    sP = np.linalg.solve(KTK, KTy)
    #sP,residual, rank,singular_vals = np.linalg.lstsq ( K, y )
    #Uncertainty
    #if unc:
    #    inv = np.linalg.inv ( KTK )
    #    (fccUnc, a0Unc, a1Unc ) = \
    #        inv.diagonal().squeeze()
    #else:
    #    (fccUnc, a0Unc, a1Unc ) = -998, -998,-998
    # get indiv. elements
    fcc = -sP[2]
    a0 = sP[0]/fcc
    a1 = sP[1]/fcc
    sBurn = a0 + lk*a1
    sFWD = pre_fire*(1-fcc) + fcc*sBurn
    #import pdb; pdb.set_trace()
    return sP, fcc, a0, a1, KTK, K, CInv, sFWD


"""
the unburnt filter is farmed out so it
could be optimised for alpha
"""
def J(alpha, x0U, c0U, obs, H, ret=False):
    """
    """
    xT = x0U
    XX = (c0U !=0).astype(float)
    # figure out what to change it
    # by
    qI = 1/alpha
    qq = np.array([ [qI, qI, qI ] for i in xrange(7)]).flatten()
    np.fill_diagonal(XX, qq)
    Pt = XX * c0U
    S = np.linalg.inv(H.dot(Pt).dot(H.T) + C_obs)
    Kt = Pt.dot(H.T).dot(S)
    # update prediction
    xtB = x0U + Kt.dot(obs - H.dot(x0U))
    # update unc
    PtB = (I - Kt.dot(H)).dot(Pt)
    # get residual
    e = (H.dot(xtB) - obs)**2
    res = e.sum()
    ATA = H.dot(PtB).dot(H.T)
    expon = (H.dot(xtB) - obs).dot(np.linalg.inv(ATA)).dot(H.dot(xtB) - obs)
    c = (2*np.pi)**(7/2.0)
    b = np.sqrt( np.linalg.det(ATA) )
    const = 1/(c * b)
    ProbB = np.exp( -0.5 * expon)

    thh, fcc, a0, a1, KTCinvK, K, CInv, xFwd = fccModel(x0U[::3], xtB[::3],
                             np.diag(c0U)[::3], np.diag(PtB)[::3], unc=True)
    theta = np.array([fcc, a0,a1])
    expon = (theta-theta_mu).T.dot(C_thetai).dot(theta-theta_mu)
    c = (2*np.pi)**(3/2.0)
    b = np.sqrt( np.linalg.det(C_theta) )
    const = 1/(c * b)
    Prior =   np.exp( -0.5 * expon)
    J = -np.log(Prior )#-np.log(alpha)
    if alpha <= 0:
        J =  1e10
    if alpha >= 1:
        J =  1e10
    if ret:
        # return all params
        return xtB, PtB, thh, fcc, a0, a1,  KTCinvK, K, CInv, xFwd
    return J



"""
stuff we'll use
"""
C_obs  = np.diag([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])
C_obsi = np.linalg.inv(C_obs)
C_obsDiag = np.diag(C_obs)
"""
Burnt surfaces 'model'
"""
theta_mu = np.array([1, 0.05, 0.15])
C_theta = np.diag([0.5, 0.1, 0.1])
C_thetai = np.linalg.inv(C_theta)
I = np.eye(21)
bCTHETA_CONST = np.sqrt( np.linalg.det(C_theta) )


def KalmanDay(x0, c0, obs, qa, Ross, Li):
    """
    daily kalman filter
    """
    alphaU = 0.94
    alphaU2 = 0.97
    alphaB = 0.01
    if qa:
        """
        remake H matrix
        """
        ha = np.array([1,  Ross, Li])
        H = np.zeros((21, 7))
        for b in xrange(7):
            H[3*b:(3*b+3), b]=ha
        H = H.T
        #
        """
        *-- solve unburnt filter --*
        """
        XX = (c0 !=0).astype(float)
        xtU = x0
        qI = 1/alphaU
        qq = np.array([ [1/alphaU, 1/alphaU2, 1/alphaU2 ] for i in xrange(7)]).flatten()
        np.fill_diagonal(XX, qq)
        Pt = XX * c0
        #S = np.linalg.inv(H.dot(Pt).dot(H.T) + C_obs)
        # speed up:
        S = np.diag(1/(np.diag(H.dot(Pt).dot(H.T)) + C_obsDiag))
        Kt = Pt.dot(H.T).dot(S)
        # update prediction
        xtU = x0 + Kt.dot(obs - H.dot(x0))
        # update unc
        PtU = (I - Kt.dot(H)).dot(Pt)
        """
        evaluate likelihood instead
        """
        ATA = H.dot(PtU).dot(H.T)
        ATAInv = np.diag(1/np.diag(ATA))
        expon = (H.dot(xtU) - obs).dot(ATAInv).dot(H.dot(xtU) - obs)
        #c = (2*np.pi)**(7/2.0)
        #b = np.sqrt( np.linalg.det(ATA) )
        #const = 1/(c * b)
        ProbU =  np.exp( -0.5 * expon)

        """
        *-- solve burnt filter --*
        """
        xT = xtU
        XX = (PtU !=0).astype(float)
        # figure out what to change it
        # by
        qI = 1/alphaB
        qq = np.array([ [qI, qI, qI ] for i in xrange(7)]).flatten()
        np.fill_diagonal(XX, qq)
        Pt = XX * PtU
        #S = np.linalg.inv(H.dot(Pt).dot(H.T) + C_obs)
        # speedup
        S = np.diag(1/(np.diag(H.dot(Pt).dot(H.T)) + C_obsDiag))
        Kt = Pt.dot(H.T).dot(S)
        # update prediction
        xtB = xtU + Kt.dot(obs - H.dot(xtU))
        # update unc
        PtB = (I - Kt.dot(H)).dot(Pt)
        """
        *-- run fcc model --*
        """
        thh, fcc, a0, a1, KTCinvK, K, CInv, xFwd = fccModel(xtU[::3], xtB[::3],
                                 np.diag(PtU)[::3], np.diag(PtB)[::3], unc=True)
        """
        evaluate likelihood instead
        """
        ATA = H.dot(PtB).dot(H.T)
        ATAInv = np.diag(1/np.diag(ATA))
        expon = (H.dot(xtB) - obs).dot(ATAInv).dot(H.dot(xtB) - obs)
        #c = (2*np.pi)**(7/2.0)
        #b = np.sqrt( np.linalg.det(ATA) )
        #const = 1/(c * b)
        ProbB =  np.exp( -0.5 * expon)
        """
        *-- evaluate quality of fcc --*
        """
        #KTCKInv = np.linalg.inv(KTCinvK)
        #HH = K.dot(KTCKInv).dot(K.T)
        #C = np.linalg.inv(CInv)
        #C = np.diag(1/np.diag(CInv))
        #((K.dot(thh) - dB[::3])**2/pp).sum()
        #dB = xtB[::3] - xtU[::3]
        #expon = (K.dot(thh) - dB).dot(CInv).dot(K.dot(thh) - dB)
        #c = (2*np.pi)**(7/2.0)
        #b = np.sqrt( np.linalg.det(C) )
        #const = 1/(c * b)
        fcc_Like = 0 # np.exp( -0.5 * expon)
        """
        *-- And add the prior for theta --*
        """
        theta = np.array([fcc, a0,a1])
        expon = (theta-theta_mu).T.dot(C_thetai).dot(theta-theta_mu)
        c = (2*np.pi)**(3/2.0)
        b = bCTHETA_CONST #np.sqrt( np.linalg.det(C_theta) )
        const = 1/(c * b)
        Prior =  const * np.exp( -0.5 * expon)
        """
        *- estimate B2 after --*
        """
    else:
        """
        No observation
        Propagate the unburnt filter
        and don't estimate the burnt filter
        """
        XX = (c0 !=0).astype(float)
        xtU = x0
        qI = 1/alphaU
        qq = np.array([ [qI, qI, qI ] for i in xrange(7)]).flatten()
        np.fill_diagonal(XX, qq)
        PtU = XX * c0
        # make fill theta
        fcc, a0, a1 = -999., -999., -999.
        # and others
        ProbB, ProbU, Prior, fcc_Like =  -999., -999., -999., -999.
    """
    Return what's needed
    """
    return ProbB, ProbU, Prior, fcc_Like, xtU, PtU, fcc, a0, a1
