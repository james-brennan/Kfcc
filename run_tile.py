"""
run_algorithms.py

Script to load data and run other algorithms
over
"""
import sys
import gdal
import glob

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
    QA_mask = land & clear  #& aerosol #& noCirrus
    return QA_mask

def vrt_loader(tile, date, xmin, ymin, xmax, ymax):
    vrt_dir = '/home/users/jbrennan01/mod09_vrts/'
    dire = vrt_dir+tile+'/'
    # get the right band
    # load it
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
    datas['sza'] = []
    datas['saa'] = []
    datas['vza'] = []
    datas['vaa'] = []



    for date in dates:
        data = vrt_loader(tile, date, xmin, ymin, xmax, ymax)
        print date
        datas['qa'].append(data['qa'])
        datas['sza'].append(data['sza'])
        datas['saa'].append(data['saa'])
        datas['vza'].append(data['vza'])
        datas['vaa'].append(data['vaa'])
        datas['date'].append(data['dates'])
        refl = np.stack(([data['brdf%i' % b] for b in xrange(1,8)]))
        refl = np.swapaxes(refl, 0,1 )
        datas['refl'].append(refl)
    # fix structure
    datas['refl'] = np.vstack(datas['refl'])
    datas['qa'] = np.vstack(datas['qa'])
    datas['sza'] = np.vstack(datas['sza'])
    datas['saa'] = np.vstack(datas['saa'])
    datas['vza'] = np.vstack(datas['vza'])
    datas['vaa'] = np.vstack(datas['vaa'])
    datas['date'] = np.hstack(datas['date'])
    datas['refl'] = datas['refl'].astype(float)
    datas['refl']*= 0.0001
    # do proper qa
    datas['qa'] = apply_QA(datas['qa'])
    # change to masked array
    datas['refl'] = np.ma.masked_array(datas['refl'],
                    ~np.swapaxes(np.stack([datas['qa']]*7), 0, 1))
    # sort out others
    datas['raa'] = ( datas['vaa']*0.01 - datas['saa']*0.01).astype( np.float32 )
    datas['vza'] = ( datas['vza']*0.01 ).astype ( np.float32 )
    datas['sza'] = ( datas['sza']*0.01 ).astype ( np.float32 )
    return datas


if __name__ == "__main__":


    """
    Figure out what data to load
    """

    tile = "h30v10"
    beginning = datetime.datetime.strptime("20080101", "%Y%m%d")
    ending = datetime.datetime.strptime("20080501", "%Y%m%d")


    """
    create outputs
    """





    """
    Load a data chunk:
    """
    for x0 in xrange(0, 2400, 256):
        for y0 in xrange(0, 2400, 256):
            x1 = x0 + 256
            y1 = y0 + 256
            x1 = np.minimum(x1, 2400)
            y1 = np.minimum(y1, 2400)
            xs = x1-x0
            ys = y1-y0
            data = LoadData(tile, beginning, ending, x0, x1, y0, y1)
            refl = data['refl']
            qa = data['qa']
            sza = data['sza']
            vza = data['vza']
            raa = data['raa']
            dates = data['dates']
            """
            Just use unique obs
            """
            _, idx = np.unique(dates, return_index=True)
            qa = qa[idx]
            refl = refl[idx]
            vza = vza[idx]
            sza = sza[idx]
            raa = raa[idx]
            dates = dates[idx]

            """
            make storage arrays
            """

            """
            Process each pixel
            """
            for x in xrange(qa[0].shape[0]):
                for y in xrange(qa[0].shape[1]):

                    kerns = Kernels(vza[:, x,y], sza[:, x,y], raa[:, x, y],
                        LiType='Sparse', doIntegrals=False,
                        normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                        RossType='Thick',nbar=0.0)

                    k = Kalman(doy, qa[:, x, y], refl[:, :, x, y], kerns)
                    p0 = np.log(1e-2*np.ones(7))
                    x, c, f = k._solve(p0)

"""
Run algorithms and save results
"""
