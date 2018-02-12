from ioKfcc import *
from kernels import *


x0 = 2000
y0 = 2000
beginning = datetime.datetime(2008, 4, 12)
ending = beginning + datetime.timedelta(days=20)

data = LoadData("h30v10", beginning, ending, 2000,2256, 2000, 2256)


tile = 'h30v10'
inputdir = '/group_workspaces/cems2/nceo_generic/users/jbrennan01/Kfcc/x0s/'
f = h5py.File( inputdir+'/'+'%s.hdf5' % (tile),'r')

x0s_ = f['state'][:, :, 2000:2256, 2000:2256]
c0s_ = f['stateunc'][ :, :, 2000:2256, 2000:2256



xx = x0s_[:, 0, 20, 20]
cc= c0s_[:, 0, 20, 20]

plt.plot(xx)
plt.plot(xx+1.96*np.sqrt(cc))
plt.plot(xx-1.96*np.sqrt(cc))

C_obs  = (([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003]))

plt.plot(xx + 1.96*np.sqrt(C_obs), 'r+--')
plt.plot(xx - 1.96*np.sqrt(C_obs), 'r+--')

"""
pick a pixel
to solve with
""" 

x,y = 21, 20 

qa = data['qa'][::2]


vza = data['vza'][::2]
raa = data['raa'][::2]
sza = data['sza'][::2]
refl = data['refl'][::2]

"""
fix qa for band5
"""
b5 = refl[:, 5]>0

qa = np.logical_and(qa, b5)

kerns =  Kernels(vza[:, y, x], sza[:, y, x], raa[:, y, x],
                LiType='Sparse', doIntegrals=False,
                normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                RossType='Thick',nbar=0.0)

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
now keep solving
"""

xs = []
cs = []


B = []
U = []
P = []

for t in xrange(20):
    ross, li = kerns.Ross[t,], kerns.Li[t,]
    inn = _x0, C0, refl[t, :, y, x], qa[t, y, x], ross, li
    out = KalmanDay(*inn)
    ProbB, ProbU, Prior, fcc_Like, xT, Pt, fcc, a0, a1 = out
    xs.append(xT)
    cs.append(Pt)
    B.append(ProbB)
    U.append(ProbU)
    P.append(fcc_Like)


xs = np.array(xs)
cs = np.array(cs)


B = np.array(B)
U = np.array(U)
P = np.array(P)


inn = _x0, C0, refl[t, :, y, x], True, ross, li
out = KalmanDay(*inn)
ProbB, ProbU, Prior, fcc_Like, xT, Pt, fcc, a0, a1 = out
