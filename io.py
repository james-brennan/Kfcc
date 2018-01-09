"""
run_algorithms.py

Script to load data and run other algorithms
over
"""
import sys
import gdal
import glob
import numpy as np
import glob, pickle
import sys
import gdal

"""
    QA functions
"""
def bitMask(qa, bitStart, bitLength, bitString="00"):
    """
    makes mask for a particular part of the modis bit string
    "inspired" from pymasker
    """
    lenstr = ''
    for i in range(bitLength):
        lenstr += '1'
    bitlen = int(lenstr, 2)

    if type(bitString) == str:
        value = int(bitString, 2)

    posValue = bitlen << bitStart
    conValue = value << bitStart
    mask = (qa & posValue) == conValue
    return mask

def apply_QA(qa):
    """
    make a QA mask
    """
    """
    use the state (1km) information to mask land/water and clouds etc
    """
    clear = bitMask(qa, bitStart=0, bitLength=2, bitString='00')
    assumedClear = bitMask(qa, bitStart=0, bitLength=2, bitString='11')
    clear = clear | assumedClear
    # check we are looking at just land pixels
    land = bitMask(qa, bitStart=3, bitLength=3, bitString='001')
    # check for cloud shadows
    noShadow = bitMask(qa, bitStart=2, bitLength=1, bitString='0')
    """
    Return overall qa mask
    """
    QA_mask = land & clear & noShadow# & noCirrus
    return QA_mask

def vrt_loader(tile, date, xmin, ymin, xmax, ymax):
    vrt_dir = '/home/users/jbrennan01/mod09_vrts/'
    dire = vrt_dir+tile+'/'
    # get the right band
    # load it
    yr = date.year
    xsize = xmax-xmin
    ysize = ymax-ymin
    data = {}
    files = ["brdf_%s_%s_b01.vrt" % (yr, tile ), "brdf_%s_%s_b02.vrt" % (yr, tile),
             "brdf_%s_%s_b03.vrt" % (yr, tile), "brdf_%s_%s_b04.vrt" % (yr, tile),
             "brdf_%s_%s_b05.vrt" % (yr, tile), "brdf_%s_%s_b06.vrt" % (yr, tile),
             "brdf_%s_%s_b07.vrt" % (yr, tile),
             "statekm_%s_%s.vrt" % (yr, tile),
             "SensorAzimuth_%s_%s.vrt" % (yr, tile),
             "SensorZenith_%s_%s.vrt" % (yr, tile),
             "SolarAzimuth_%s_%s.vrt" % (yr, tile),
             "SolarZenith_%s_%s.vrt" % (yr, tile),]
    dNames = ['brdf1', 'brdf2', 'brdf3', 'brdf4', 'brdf5',
              'brdf6', 'brdf7', 'qa', 'vaa', 'vza', 'saa', 'sza', ]
    qainfo = gdal.Open(dire+"statekm_%s_%s.vrt" % (yr, tile))
    doy = np.array([int(qainfo.GetRasterBand(b+1).GetMetadataItem("DoY")) for b in xrange(qainfo.RasterCount)])
    year_doy = np.array([int(qainfo.GetRasterBand(b+1).GetMetadataItem("Year")) for b in xrange(qainfo.RasterCount)])
    dates = np.array([datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1) for year, days in zip(year_doy, doy)])

    # select correct date

    idx = np.where(dates==date)[0]+1 # add 1 for GDAL
    # load these bands
    for nm, p in zip(dNames, files):
        datastack = []
        for band in idx:
            pp = gdal.Open(dire+p)
            data_p = pp.GetRasterBand(band)
            data_ = data_p.ReadAsArray(xoff=xmin, yoff=ymin, win_xsize=xsize, win_ysize=ysize)
            datastack.append(data_)
            #print nm, p, band
        data[nm]=np.array(datastack)
    data['dates'] = dates[idx-1]
    return data



def LoadData(tile, beginning, ending, xmin, xmax, ymin, ymax):
    """
    Loads the modis refl and active fires
    for the time-span
    """
    # figure what dates we need
    ndays = (ending-beginning).days
    dates = np.array([beginning + datetime.timedelta(days=x) for x in range(0, ndays)])
    """
    for a date
    load the necessary band
    """
    datas = {}
    # add stuff
    datas['qa'] = []
    datas['refl'] = []
    datas['date'] = []
    datas['vza'] = []
    datas['sza'] = []
    datas['raa'] = []




    ida = 0
    n = len(dates)
    for date in dates:
        data = vrt_loader(tile, date, xmin, ymin, xmax, ymax)
        datas['qa'].append(data['qa'])
        datas['date'].append(data['dates'])
        refl = np.stack(([data['brdf%i' % b] for b in xrange(1,8)]))
        refl = np.swapaxes(refl, 0,1 )
        refl = refl.astype(float)
        refl *= 0.0001
        """
        fix mask errors
        due to band6
        """
        newMask = np.logical_or(~data['qa'], refl[:, 5]==0)
        newMask = np.swapaxes(np.stack([newMask]*7), 0, 1)
        refl = np.ma.masked_array(refl,newMask)
        datas['refl'].append( refl )
        datas['vza'].append( data['vza']*0.01 )
        datas['sza'].append( data['sza']*0.01 )
        datas['raa'].append( (data['vaa']*0.01 - data['saa']*0.01).astype( np.float32 ))
        logging.info("Loaded data for %i / %i" % (ida, n))
        ida += 1
    # fix structure
    import pdb; pdb.set_trace()
    datas['refl'] = np.vstack(datas['refl'])
    datas['qa'] = np.vstack(datas['qa'])
    datas['vza'] = np.vstack(datas['vza']).astype(float)
    datas['sza'] = np.vstack(datas['sza']).astype(float)
    datas['raa'] = np.vstack(datas['raa']).astype(float)
    datas['date'] = np.hstack(datas['date'])
    # do proper qa
    datas['qa'] = apply_QA(datas['qa'])
    # change to masked array
    datas['refl'] = np.ma.masked_array(datas['refl'],
                    ~np.swapaxes(np.stack([datas['qa']]*7), 0, 1))
    """
    Now also load the active fires
    Load all and filter by date
    """
    year = 2008
    af_dir = '/group_workspaces/cems2/nceo_generic/users/jbrennan01/RRGlob/MCD14/'
    ffiles = glob.glob(af_dir+"MOD14*%i*%s*" % (year, tile))
    ffiles.sort()
    active_fires = np.zeros((366, 1200, 1200), dtype=np.bool)
    tmp = 'HDF4_EOS:EOS_GRID:"%s":MODIS_Grid_Daily_Fire:FireMask'
    for f in ffiles:
        aa = gdal.Open(tmp%f).ReadAsArray()
        start = int(f.split(".")[1][-3:])
        end = aa.shape[0] + start
        if end>365:
            end = 365
            aa = aa[:end-start]
        active_fires[start:end]=aa>7
    datas['active_fires'] = active_fires
    return datas
