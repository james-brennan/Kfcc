"""
solve_day.py

code for running Kalman filter for a day
"""
from ioKfcc import *
import h5py
import numpy as np
import datetime
import logging
from kernels import *
import scipy.linalg
import KalDay
import matplotlib.pyplot as plt

cc = 1.0
cI = np.eye(21) * cc

qIso = np.zeros(21)
qIso[::3]=([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])
qIso = np.diag(qIso)



if __name__ == "__main__":
    logging.basicConfig(filename="log.log",
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    """
    Load initial conditions
    """
    tile = 'h30v10'
    inputdir = '/group_workspaces/cems2/nceo_generic/users/jbrennan01/Kfcc/x0s/'
    f = h5py.File( inputdir+'/'+'%s.hdf5' % (tile),'r')

    # run for
    NDAYS = 3

    """
    make some storage file
    """
    output_dir = '/group_workspaces/cems2/nceo_generic/users/jbrennan01/Kfcc/sols/'
    outfile = h5py.File( output_dir+'/'+'%s.hdf5' % (tile),'w')
    B = outfile.create_dataset("B", (NDAYS, 2400, 2400), dtype="f4", chunks=(NDAYS, 256, 256))
    U = outfile.create_dataset("U", (NDAYS, 2400, 2400), dtype="f4", chunks=(NDAYS, 256, 256))
    P = outfile.create_dataset("P", (NDAYS, 2400, 2400), dtype="f4", chunks=(NDAYS, 256, 256))
    L = outfile.create_dataset("L", (NDAYS, 2400, 2400), dtype="f4", chunks=(NDAYS, 256, 256))
    theta = outfile.create_dataset("theta", (NDAYS, 3, 2400, 2400), dtype="f4", chunks=(NDAYS, 3, 256, 256))

    # and for now store evols
    ks = outfile.create_dataset("state", (NDAYS, 7, 3, 2400, 2400), dtype="f4", chunks=(NDAYS, 7, 3, 256, 256))
    ks_unc = outfile.create_dataset("unc", (NDAYS, 7, 9, 2400, 2400), dtype="f4", chunks=(NDAYS, 7, 9, 256, 256))


    """
    Process days
    """
    beginning = datetime.datetime(2008, 4, 12)
    ending = beginning + datetime.timedelta(days=NDAYS)

    """
    process in chunks to save memory
    """
    for x0 in xrange(2000, 2400, 256):
        for y0 in xrange(2000, 2400, 256):
            x1 = x0 + 256
            y1 = y0 + 256
            x1 = np.minimum(x1, 2400)
            y1 = np.minimum(y1, 2400)
            xs = x1-x0
            ys = y1-y0
            logging.info("Processing %i %i" % (y0, x0))
            print("Processing %i %i" % (y0, x0))

            """
            make some temp storage
            """
            sB = -998*np.ones((NDAYS, ys, xs))
            sU = -998*np.ones((NDAYS, ys, xs))
            sTheta = -998*np.ones((NDAYS, 3, ys, xs))
            sP = -998*np.ones((NDAYS, ys, xs))
            sL = -998*np.ones((NDAYS, ys, xs))
            sK = -998*np.ones((NDAYS, 7, 3, ys, xs))
            sCov =  -998*np.ones((NDAYS, 7, 9, ys, xs))
            """
            make somewhere to store results
            for the previous day
            """
            x0s = np.zeros((7,3, ys, xs))
            c0s = np.zeros((7,9, ys, xs))
            """
            sort out date stuff
            """
            date = beginning
            iday = 0
            while date < ending:
                date1 = date + datetime.timedelta(days=1)
                data = LoadData(tile, date, date1, x0, x1, y0, y1)
                """
                For each pixel derive the initial condition
                using Kalman smoother/Regularisation

                -- Choose AQUA due to Terra Band 5 issue
                """
                # find best ob
                #idx = np.argmax([data['qa'][0].sum(), data['qa'][1].sum()])
                #idx = 1 # not sure how to deal with dodgy terra
                idx = np.where(data['sensor']=="TERRA")[0][0]
                #print idx, data['sensor'][idx]
                vza = data['vza'][idx]
                raa = data['raa'][idx]
                sza = data['sza'][idx]
                refl = data['refl'][idx]
                qa = data['qa'][idx]

                """
                Fix QA for band5
                """
                b5 = refl[5]>0
                qa = np.logical_and(qa, b5)
                """
                Load initial conditions
                """
                x0s_ = f['state'][:, :, y0:y1, x0:x1]
                c0s_ = f['stateunc'][ :, :, y0:y1, x0:x1]
                # Then also somewhere to store them in real_time

                #import pdb; pdb.set_trace()

                """
                get kernels
                """
                kerns =  Kernels(vza, sza, raa,
                                LiType='Sparse', doIntegrals=False,
                                normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                                RossType='Thick',nbar=0.0)
                kerns.Ross = kerns.Ross.reshape((ys, xs))
                kerns.Li = kerns.Li.reshape((ys, xs))
                """
                Find out whether land or water based on the initial
                conditions
                """
                LAND = x0s_[0, 0]!=-999
                for y in xrange(ys):
                    for x in xrange(xs):
                        """
                        if this is the first date get the initial estimate
                        """
                        if LAND[y, x]:
                            # process
                            #print y, x
                            if date == beginning:
                                _x0 = x0s_[:, :, y, x].reshape(21)
                                _c0 = c0s_[:, :, y, x]

                                """
                                also increase variance a of the iso
                                """
                                #_c0 *= 3
                                # reshape it
                                arrs = [_c0[band].reshape((3,3)) for band in xrange(7)]
                                C0 = scipy.linalg.block_diag(*arrs)
                                C0 += qIso
                                """
                                Increase covariance to better trust
                                observations
                                """
                                #C0 = cI * C0

                                """
                                i think we also need to fill the other x0
                                so that if it's the first time theres
                                an observation we use the starting
                                condition
                                BUT of course overwrite if not...
                                """
                                x0s = x0s_
                                c0s = c0s_
                            else:
                                """
                                we've already run the filter so have
                                the t-1 estimates
                                so let's retrieve them
                                """
                                _x0 = x0s[:, :, y, x].reshape(21)
                                _c0 = c0s[:, :, y, x]
                                # reshape it
                                arrs = [_c0[band].reshape((3,3)) for band in xrange(7)]
                                C0 = scipy.linalg.block_diag(*arrs)
                                #import pdb; pdb.set_trace()
                                if np.all(x0s[:, :, y, x]==0):
                                    # something weird
                                    # repeat initial
                                    _x0 = x0s_[:, :, y, x].reshape(21)
                                    _c0 = c0s_[:, :, y, x]
                                    # reshape it
                                    arrs = [_c0[band].reshape((3,3)) for band in xrange(7)]
                                    C0 = scipy.linalg.block_diag(*arrs)
                            """
                            Run kalman filter over every pixel in this chunk
                            """
                            ross, li = kerns.Ross[y, x], kerns.Li[y, x]
                            inn = _x0, C0, refl[:, y, x], qa[y, x], ross, li
                            try:
                                out = KalDay.KalmanDay(*inn)
                                ProbB, ProbU, Prior, fcc_Like, xT, Pt, fcc, a0, a1 = out
                            except:
                                # oops ermmm
                                # prop again?
                                ProbB, ProbU, Prior, fcc_Like, fcc, a0, a1 = np.ones(7)*-999.
                                xT = _x0
                                Pt = C0
                            """
                            Save
                            to temp
                            """
                            sB[ iday, y,x]=ProbB
                            sU[ iday, y,x]=ProbU
                            sP[ iday, y,x]=Prior
                            sL[ iday, y,x]=fcc_Like
                            sTheta[ iday, :, y,x]=(fcc, a0, a1)



                            """
                            and store state for next day
                            """
                            x0s[:, :, y, x] = xT.reshape((7, 3))
                            _cc = np.array([Pt[3*b:(3+3*b), 3*b:(3+3*b)].flatten() for b in xrange(7)])
                            c0s[:, :, y, x] = _cc
                            #if qa[y, x] and ProbB == -999:
                            #if qa[y, x]:
                            #    print iday, y, x, qa[y, x], ProbB
                            sK[iday, :, :, y, x] = xT.reshape((7, 3))
                            sCov[iday, :, :, y, x] =_cc
                        else:
                            """
                            not a land pixel pass
                            """
                            pass
                # increment date
                date += datetime.timedelta(days=1)
                iday += 1
                import pdb; pdb.set_trace()
            """
            write to outputs
            """
            #import pdb; pdb.set_trace()
            B[:, y0:y1, x0:x1] = sB
            U[:,  y0:y1, x0:x1] = sU
            P[:,  y0:y1, x0:x1] = sP
            L[:,  y0:y1, x0:x1] = sL
            theta[:, :, y0:y1, x0:x1]=sTheta
            ks[:, :, :,  y0:y1, x0:x1]=sK
            ks_unc[:, :, :,  y0:y1, x0:x1]=sCov
            np.save("theta.npy", sTheta)
            np.save("U.npy", sU)
            np.save("B.npy", sB)
            np.save("P.npy", sP)
            np.save("L.npy", sL)
