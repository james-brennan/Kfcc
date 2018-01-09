"""
initial.py

Derive intial conditions using a Kalman smoother

-- Actually uses regularisation for now
"""
import logging
from io import *

if __name__ == "__main__":
    """
    Load data
    """
    tile = 'h30v10'
    year = 2008
    log_dir = '/home/users/jbrennan01/DATA2/Kfcc/logs/'
    # add filemode="w" to overwrite
    logging.basicConfig(filename=log_dir+"%s_%i.log" % (tile, year), level=logging.INFO)
    logging.info("Doing initial conditions")


    """
    Do in chunks
    """
    beginning = datetime.datetime(2008, 1, 1)
    ending = datetime.datetime(2008, 3, 1)

    for x0 in xrange(0, 2400, 256):
        for y0 in xrange(0, 2400, 256):
            x1 = x0 + 256
            y1 = y0 + 256
            x1 = np.minimum(x1, 2400)
            y1 = np.minimum(y1, 2400)
            xs = x1-x0
            ys = y1-y0
            data = LoadData(tile, beginning, ending, x0, x1, y0, y1)
            break

            """
            For each pixel derive the initial condition
            using Kalman smoother/Regularisation
            """
            vza = data['vza']
            raa = data['raa']
            sza = data['sza']
            refl = data['refl']
            qa = data['qa']
            x = 1
            y = 1
            date = data['date']

            kerns  =  Kernels( vza[:, x, y], sza[:,x,y], raa[:, x, y], \
                        LiType='Sparse', doIntegrals=False, \
                        normalise=1, RecipFlag=True, RossHS=True, MODISSPARSE=True, \
                        RossType='Thick' )

    r = refl[:,:, x,y]
    q = qa[:, x, y]
    r = r.data

    t = regularisation(date, r, q, vza[:, x, y], sza[:,x,y], raa[:, x, y], regMatrix=None)

    p = t.solve(band=1, alpha=1e1, calculate_uncs=True)




