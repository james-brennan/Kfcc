"""
initial.py

Derive intial conditions using normal brdf corr
now just using a kalman filter...
"""
import logging
from ioKfcc import *
import datetime
import numpy as np
from kernels import *
import copy
import h5py


band_uncs = np.array([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])

if __name__ == "__main__":
    """
    Load data
    """
    tile = 'h19v10'
    year = 2008
    log_dir = '/home/users/jbrennan01/DATA2/Kfcc/logs/'
    # add filemode="w" to overwrite
    logging.basicConfig(filename=log_dir+"%s_%i.log" % (tile, year), level=logging.INFO)
    logging.info("Doing initial conditions")

    """
    make output storage
    """
    output_dir = '/group_workspaces/cems2/nceo_generic/users/jbrennan01/Kfcc/x0s/'
    f = h5py.File( output_dir+'/'+'%s.hdf5' % (tile),'w')
    state = f.create_dataset("state", (7, 3, 2400, 2400), dtype="f4", chunks=(7, 3, 256, 256))
    stateunc = f.create_dataset("stateunc", (7, 9, 2400, 2400), dtype="f4", chunks=(7, 9, 256, 256))
    """
    Do in chunks
    """
    beginning = datetime.datetime(2008, 2, 1)
    ending = datetime.datetime(2008, 4, 10)
    for x0 in xrange(0, 2400, 256):
        for y0 in xrange(0, 2400, 256):
            x1 = x0 + 256
            y1 = y0 + 256
            x1 = np.minimum(x1, 2400)
            y1 = np.minimum(y1, 2400)
            xs = x1-x0
            ys = y1-y0
            data = LoadData(tile, beginning, ending, x0, x1, y0, y1)
            """
            For each pixel derive the initial condition
            using Kalman smoother/Regularisation
            """
            vza = data['vza']
            raa = data['raa']
            sza = data['sza']
            refl = data['refl']
            qa = data['qa']
            sensor = data['sensor']
            store_state = np.zeros((7, 3, ys, xs))
            store_unc = np.zeros((7, 9, ys, xs))
            #sensQ = sensor == "AQUA"
            for x in xrange(xs):
                for y in xrange(ys):
                    date = data['date']
                    r = refl[:,:, y,x]
                    q = qa[:, y, x]
                    #import pdb; pdb.set_trace()
                    #q = np.logical_and(q, sensQ)
                    """
                    check more than 5 observations
                    """
                    #print x, y
                    if q.sum() < 5:
                        # not enough obs
                            store_state[ :, :, y,x]=-999
                            store_unc[ :, :, y,x]=-999
                    else:
                        try:
                            #import pdb; pdb.set_trace()
                            """
                            do BRDF inversion
                            """
                            kerns =  Kernels(vza[:, y, x][q],
                                            sza[:, y, x][q],
                                            raa[:, y, x][q],
                                            LiType='Sparse', doIntegrals=False, \
                                            normalise=1, RecipFlag=True, RossHS=True, MODISSPARSE=True, \
                                            RossType='Thick', nbar=0.0)
                            """
                            solve system for each band
                            """
                            C =  np.diag( np.ones(q.sum()) )
                            for band in xrange(7):
                                yr = r[q, band]
                                A = np.vstack((kerns.Isotropic, kerns.Ross, kerns.Li)).T
                                np.fill_diagonal(C, (1/band_uncs[band]) )
                                ATCA = A.T.dot(C).dot(A)
                                ATCy = A.T.dot(C).dot(yr)
                                #ATCA = A.T.dot(A)
                                #ATCy = A.T.dot(yr)
                                sol = np.linalg.solve(ATCA,ATCy )
                                # get unc
                                II = np.linalg.inv(ATCA)
                                store_state[ band, :, y,x]=sol
                                store_unc[band, :, y,x]=II.flatten()
                                yhat = A.dot(sol)
                            #import pdb; pdb.set_trace()
                        except:
                            # didnt work make all -998
                            store_state[ :, :, y,x]=-998
                            store_unc[ :, :, y,x]=-998
            """
            save
            """
            #import pdb; pdb.set_trace()
            state[:, :, y0:y1, x0:x1] = store_state
            stateunc[ :, :, y0:y1, x0:x1] = store_unc
            # save to check atleast one
            np.save(output_dir+"state", store_state)
            np.save(output_dir+"unc", store_state)
            logging.info("Done %i %i %i %i" % (x0, y0, x1, y1))
    f.close()
