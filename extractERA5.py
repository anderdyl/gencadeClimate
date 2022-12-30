import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


slpMemory = False
lonLeft = 250
lonRight = 350
latBot = 10
latTop = 50
startTime = [1979,1,1]
avgTime = 24
resolution = 1
endTime = [1979,3,31]
slpPath = '/users/dylananderson/documents/data/prmsl/'


dt = datetime.date(startTime[0], startTime[1], startTime[2])
end = datetime.date(endTime[0], endTime[1], endTime[2])
step = relativedelta(months=1)
extractTime = []
while dt < end:
    extractTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


import cdsapi
import xarray as xr
from urllib.request import urlopen

counter = 0
for hh in range(len(extractTime)):

    # start the client
    cds = cdsapi.Client()
    # dataset you want to read
    dataset = 'reanalysis-era5-single-levels'
    # flag to download data
    download_flag = False
    # api parameters
    params = {
        "format": "netcdf",
        "product_type": "reanalysis",
        "variable": "mean_sea_level_pressure",
        'year':[str(extractTime[hh].year)],
        'month':[str(extractTime[hh].month),],
        'day': ['01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',],
        "time": [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        "area": [latTop, lonLeft, latBot, lonRight],
        }
    # retrieves the path to the file
    fl = cds.retrieve(dataset, params)
    # download the file
    if download_flag:
        fl.download("./output.nc")
    # load into memory
    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())

    # are we averaging to a shorter time window?
    m, n, p = np.shape(ds.msl)
    slp_ = np.zeros((n * p, m))
    grd_ = np.zeros((n * p, m))

    for mmm in range(m):
        slp_[:, mmm] = ds.msl[mmm, :, :].values.flatten()
        vgrad = np.gradient(ds.msl[mmm, :, :].values)
        grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
    # slp_ = slp.reshape(n*m,p)

    # are we averaging to a shorter time window?
    if avgTime == 0:
        print('returning hourly values')
        slpAvg = slp_
        grdAvg = grd_
        datesAvg = ds.time.values
    else:
        numWindows = int(len(ds.time.values) / avgTime)
        print('averaging every {} hours to {} timesteps'.format(avgTime, numWindows))
        c = 0
        datesAvg = list()
        slpAvg = np.empty((n * p, numWindows))
        grdAvg = np.empty((n * p, numWindows))
        for t in range(numWindows):
            slpAvg[:, t] = np.nanmean(slp_[:, c:c + avgTime], axis=1)
            grdAvg[:, t] = np.nanmean(grd_[:, c:c + avgTime], axis=1)
            datesAvg.append(ds.time.values[c])
            c = c + avgTime

    # are we reducing the resolution of the grid?
    if resolution == 0.25:
        print('keeping full 0.25 degree resolution')
        slpDownscaled = slpAvg
        grdDownscaled = grdAvg
        x, y = np.meshgrid(ds.longitude.values,ds.latitude.values.flatten())
        xDownscaled = x.flatten()
        yDownscaled = y.flatten()
        x2 = x
        y2 = y
    else:
        x, y = np.meshgrid(ds.longitude.values,ds.latitude.values.flatten())
        xFlat = x.flatten()
        yFlat = y.flatten()
        xRem = np.fmod(xFlat, resolution)
        yRem = np.fmod(yFlat, resolution)
        cc = 0
        for pp in range(len(xFlat)):
            if xRem[pp] == 0:
                if yRem[pp] == 0:
                    if cc == 0:
                        ind2deg = int(pp)
                        cc = cc + 1
                    else:
                        ind2deg = np.hstack((ind2deg, int(pp)))
                        cc = cc + 1
        slpDownscaled = slpAvg[ind2deg, :]
        grdDownscaled = grdAvg[ind2deg, :]
        xDownscaled = xFlat[ind2deg]
        yDownscaled = yFlat[ind2deg]
        print('Downscaling to {} degree resolution'.format(resolution))
        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
        reshapeIndX = np.where((np.diff(xDownscaled) > 4) | (np.diff(xDownscaled) < -4))
        reshapeIndY = np.where((np.diff(yDownscaled) < 0))
        x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1))
        y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1))

    My, Mx = np.shape(y2)
    slpMem = slpDownscaled
    datesMem = datesAvg
    grdMem = grdDownscaled
    if counter == 0:
        SLPS = slpMem
        GRDS = grdMem
        DATES = datesMem
    else:
        SLPS = np.hstack((SLPS, slpMem))
        GRDS = np.hstack((GRDS, grdMem))
        DATES = np.append(DATES, datesMem)
    counter = counter + 1





# alternate world where this is handled all in xarray...

    # dsDay = ds.groupby("time.day").mean(dim='time')
    # dt = datetime.date(extractTime[hh].year, extractTime[hh].month, 1)
    # if extractTime[hh].month < 12:
    #     end = datetime.date(extractTime[hh].year, extractTime[hh].month+1, 1)
    # else:
    #     end = datetime.date(extractTime[hh].year+1, extractTime[hh].month, 1)
    # step = relativedelta(days=1)
    # dailyTime = []
    # while dt < end:
    #     dailyTime.append(dt)  # .strftime('%Y-%m-%d'))
    #     dt += step
    # dsDay['time'] = dailyTime
    # if hh == 0:
    #     df = dsDay
    # else:
    #     df = xr.concat([df,dsDay],dim='day')

