
import numpy as np
import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset
import xarray as xr

class weatherTypes():
    '''
    Class containing camera data and functions'''

    def __init__(self, **kwargs):

        self.lonLeft = kwargs.get('lonLeft', 275)
        self.lonRight = kwargs.get('lonRight',350)
        self.latBot = kwargs.get('latBot', 15)
        self.latTop = kwargs.get('latTop', 50)
        self.resolution = kwargs.get('resolution',1)
        self.avgTime = kwargs.get('avgTime',24)
        self.startTime = kwargs.get('startTime',[1979,1,1])
        self.endTime = kwargs.get('endTime',[2020,12,31])
        self.slpMemory = kwargs.get('slpMemory',False)
        self.slpPath = kwargs.get('slpPath')
        # self.lonLeft = kwargs.get('cameraID', 'c1')
        # self.lonRight = kwargs.get('rawPath')
        # self.latBottom = kwargs.get('nFrames', 1)
        # self.latTop = kwargs.get('startFrame', 0)

    def extractERA5(self,printToScreen=False):
        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        dt = date(self.startTime[0], self.startTime[1], self.startTime[2])
        end = date(self.endTime[0], self.endTime[1], self.endTime[2])
        step = relativedelta(months=1)
        extractTime = []
        while dt < end:
            extractTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        import cdsapi
        import xarray as xr
        from urllib.request import urlopen

        counter = 0
        for hh in range(len(extractTime)):
            if printToScreen == True:
                print('{}-{}'.format(extractTime[hh].year, extractTime[hh].month))
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
                'year': [str(extractTime[hh].year)],
                'month': [str(extractTime[hh].month), ],
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
                        '31', ],
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
                "area": [self.latTop, self.lonLeft, self.latBot, self.lonRight],
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
            if self.avgTime == 0:
                print('returning hourly values')
                slpAvg = slp_
                grdAvg = grd_
                datesAvg = ds.time.values
            else:
                numWindows = int(len(ds.time.values) / self.avgTime)
                print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                c = 0
                datesAvg = list()
                slpAvg = np.empty((n * p, numWindows))
                grdAvg = np.empty((n * p, numWindows))
                for t in range(numWindows):
                    slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                    grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                    datesAvg.append(ds.time.values[c])
                    c = c + self.avgTime

            # are we reducing the resolution of the grid?
            if self.resolution == 0.25:
                print('keeping full 0.25 degree resolution')
                slpDownscaled = slpAvg
                grdDownscaled = grdAvg
                x, y = np.meshgrid(ds.longitude.values, ds.latitude.values.flatten())
                xDownscaled = x.flatten()
                yDownscaled = y.flatten()
                x2 = x
                y2 = y
            else:
                x, y = np.meshgrid(ds.longitude.values, ds.latitude.values.flatten())
                xFlat = x.flatten()
                yFlat = y.flatten()
                xRem = np.fmod(xFlat, self.resolution)
                yRem = np.fmod(yFlat, self.resolution)
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
                print('Downscaling to {} degree resolution'.format(self.resolution))
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

        print('Extracted until {}-{}'.format(year2, month2))

        from global_land_mask import globe
        wrapLons = np.where((x2 > 180))
        x2[wrapLons] = x2[wrapLons] - 360
        xFlat = x2.flatten()
        yFlat = y2.flatten()

        isOnLandGrid = globe.is_land(y2, x2)
        isOnLandFlat = globe.is_land(yFlat, xFlat)

        x2[wrapLons] = x2[wrapLons] + 360
        xFlat = x2.flatten()

        self.xGrid = x2
        self.yGrid = y2
        self.xFlat = xFlat
        self.yFlat = yFlat
        self.isOnLandGrid = isOnLandGrid
        self.isOnLandFlat = isOnLandFlat
        self.SLPS = SLPS
        self.GRDS = GRDS
        self.DATES = DATES
        self.Mx = Mx
        self.My = My

    def extractCFSR(self,printToScreen=False):
        '''
        This function is utilized for opening *.raw Argus files prior to a debayering task,
        and adds the

        Must call this function before calling any debayering functions
        '''
        # with open(self.rawPath, "rb") as my_file:
        #     self.fh = my_file
        #     cameraIO.readHeaderLines(self)
        #     cameraIO.readAIIIrawFullFrameWithTimeStamp(self)

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        # estelaMat = '/media/dylananderson/Elements1/ESTELA/out/NagsHead2/NagsHead2_obj.mat'

        filePaths = np.sort(os.listdir(self.slpPath))

        initFile = self.slpPath + 'prmsl.cdas1.201104.grb2.nc'

        dt = datetime(year, month, 1)
        if month2 == 12:
            end = datetime(year2 + 1, 1, 1)
        else:
            end = datetime(year2, month2 + 1, 1)
        # step = datetime.timedelta(months=1)
        step = relativedelta(months=1)
        extractTime = []
        while dt < end:
            extractTime.append(dt)
            dt += step

        data = Dataset(initFile)
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]

        pos_lon1 = np.where((lon == self.lonLeft - 2))
        pos_lon2 = np.where((lon == self.lonRight + 2))
        pos_lat2 = np.where((lat == self.latTop + 2))
        pos_lat1 = np.where((lat == self.latBot - 2))

        latitud = lat[(pos_lat2[0][0]):(pos_lat1[0][0] + 1)]
        if self.lonLeft > self.lonRight:
            longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
        else:
            longitud = lon[pos_lon1[0][0]:(pos_lon2[0][0] + 1)]
        [x, y] = np.meshgrid(longitud, latitud)

        counter = 0
        # Now need to loop through the number of files we're extracting data from
        for tt in extractTime:

            if self.slpMemory == False:
                yearExtract = tt.year
                monthExtract = tt.month

                if (yearExtract == 2011 and monthExtract >= 4) or yearExtract > 2011:
                    file = self.slpPath + 'prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                           monthExtract)
                    # file = '/users/dylananderson/documents/data/prmsl/prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                    #                                                                                        monthExtract)
                else:
                    file = self.slpPath + 'prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                          monthExtract)
                    # file = '/users/dylananderson/documents/data/prmsl/prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                    #                                                                                       monthExtract)
                data = Dataset(file)
                time = data.variables['valid_date_time']
                if printToScreen == True:

                    print('{}-{}'.format(yearExtract, monthExtract))

                # extract times and turn them into datetimes
                # years = np.empty((len(time),))
                year = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[0:4]) for i in time]
                month = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[4:6]) for i in time]
                day = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[6:8]) for i in time]
                hour = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[8:]) for i in time]
                d_vec = np.vstack((year, month, day, hour)).T
                dates = [datetime(d[0], d[1], d[2], d[3], 0, 0) for d in d_vec]

                # Extracting SLP fields, need to account for wrap around international date line
                if self.lonLeft > self.lonRight:
                    # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                    slp1 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                    slp2 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                           0:(pos_lon2[0][0] + 1)]
                    slp = np.concatenate((slp1, slp2), axis=2)
                else:
                    slp = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                          (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]

                # are we averaging to a shorter time window?
                m, n, p = np.shape(slp)
                slp_ = np.zeros((n * p, m))
                grd_ = np.zeros((n * p, m))
                for mmm in range(m):
                    slp_[:, mmm] = slp[mmm, :, :].flatten()
                    vgrad = np.gradient(slp[mmm, :, :])
                    grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
                # slp_ = slp.reshape(n*m,p)

                if self.avgTime == 0:
                    if printToScreen == True:
                        print('returning hourly values')
                    elif counter == 0:
                        print('returning hourly values')

                    slpAvg = slp_
                    grdAvg = grd_
                    datesAvg = dates
                else:
                    numWindows = int(len(time) / self.avgTime)
                    if printToScreen == True:
                        print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                    elif counter == 0:
                        print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))

                    c = 0
                    datesAvg = list()
                    slpAvg = np.empty((n * p, numWindows))
                    grdAvg = np.empty((n * p, numWindows))

                    for t in range(numWindows):
                        slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                        grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                        datesAvg.append(dates[c])
                        c = c + self.avgTime

                # are we reducing the resolution of the grid?
                if self.resolution == 0.5:
                    if printToScreen == True:
                        print('keeping full 0.5 degree resolution')
                    elif counter == 0:
                        print('keeping full 0.5 degree resolution')

                    slpDownscaled = slpAvg
                    grdDownscaled = grdAvg
                    xDownscaled = x.flatten()
                    yDownscaled = y.flatten()
                    x2 = x
                    y2 = y
                else:
                    xFlat = x.flatten()
                    yFlat = y.flatten()
                    xRem = np.fmod(xFlat, self.resolution)
                    yRem = np.fmod(yFlat, self.resolution)
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
                    if printToScreen == True:
                        print('Downscaling to {} degree resolution'.format(self.resolution))
                        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
                    elif counter == 0:
                        print('Downscaling to {} degree resolution'.format(self.resolution))
                        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
                    reshapeIndX = np.where((np.diff(xDownscaled) > 4) | (np.diff(xDownscaled) < -4))
                    reshapeIndY = np.where((np.diff(yDownscaled) < 0))
                    x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1)).filled()
                    y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1)).filled()

                My, Mx = np.shape(y2)
                slpMem = slpDownscaled
                grdMem = grdDownscaled
                datesMem = datesAvg


            else:

                for mm in range(2):
                    if mm == 0:
                        monthExtract = tt.month - 1
                        if monthExtract == 0:
                            monthExtract = 12
                            yearExtract = tt.year - 1
                        else:
                            yearExtract = tt.year
                    else:
                        yearExtract = tt.year
                        monthExtract = tt.month

                    if (yearExtract == 2011 and monthExtract >= 4) or yearExtract > 2012:
                        file = self.slpPath + 'prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                                    monthExtract)
                        # file = '/users/dylananderson/documents/data/prmsl/prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                        #                                                                                        monthExtract)
                    else:
                        file = self.slpPath + 'prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                                   monthExtract)
                        # file = '/users/dylananderson/documents/data/prmsl/prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                        #                                                                                       monthExtract)

                    data = Dataset(file)
                    time = data.variables['valid_date_time']
                    print('{}-{}'.format(yearExtract, monthExtract))

                    # extract times and turn them into datetimes
                    # years = np.empty((len(time),))
                    year = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[0:4]) for i in time]
                    month = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[4:6]) for i in time]
                    day = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[6:8]) for i in time]
                    hour = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[8:]) for i in time]
                    d_vec = np.vstack((year, month, day, hour)).T
                    dates = [datetime(d[0], d[1], d[2], d[3], 0, 0) for d in d_vec]

                    # Extracting SLP fields, need to account for wrap around international date line
                    if self.lonLeft > self.lonRight:
                        # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                        slp1 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                        slp2 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                               0:(pos_lon2[0][0] + 1)]
                        slp = np.concatenate((slp1, slp2), axis=2)
                    else:
                        slp = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                              (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]

                # are we averaging to a shorter time window?
                m, n, p = np.shape(slp)
                slp_ = np.zeros((n * p, m))
                grd_ = np.zeros((n * p, m))
                for mmm in range(m):
                    slp_[:, mmm] = slp[mmm, :, :].flatten()
                    vgrad = np.gradient(slp[mmm, :, :])
                    grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
                # slp_ = slp.reshape(n*m,p)

                if self.avgTime == 0:
                    if printToScreen == True:
                        print('returning hourly values')
                    slpAvg = slp_
                    grdAvg = grd_
                    datesAvg = dates
                else:
                    numWindows = int(len(time) / self.avgTime)
                    if printToScreen == True:
                        print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                    c = 0
                    datesAvg = list()
                    slpAvg = np.empty((n * p, numWindows))
                    grdAvg = np.empty((n * p, numWindows))
                    for t in range(numWindows):
                        slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                        grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                        datesAvg.append(dates[c])
                        c = c + self.avgTime

                # are we reducing the resolution of the grid?
                if self.resolution == 0.5:
                    if printToScreen == True:
                        print('keeping full 0.5 degree resolution')
                    slpDownscaled = slpAvg
                    grdDownscaled = grdAvg
                    xDownscaled = x.flatten()
                    yDownscaled = y.flatten()
                    x2 = x
                    y2 = y
                else:
                    xFlat = x.flatten()
                    yFlat = y.flatten()
                    xRem = np.fmod(xFlat, self.resolution)
                    yRem = np.fmod(yFlat, self.resolution)
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
                    if printToScreen == True:
                        print('Downscaling to {} degree resolution'.format(self.resolution))
                        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
                    reshapeIndX = np.where((np.diff(xDownscaled) > 4) | (np.diff(xDownscaled) < -4))
                    reshapeIndY = np.where((np.diff(yDownscaled) < 0))
                    x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0] + 1))
                    y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0] + 1))
                My, Mx = np.shape(y2)
                slpMem = slpDownscaled
                datesMem = datesAvg
                grdMem = grdDownscaled

                # dim, Ntime = np.shape(slpDownscaled)
                # GrdSlp = np.zeros((np.shape(slpDownscaled)))
                # for ttt in range(Ntime):
                #     p = slpDownscaled[:,ttt].reshape(My,Mx)
                #     dp = np.zeros((np.shape(p)))
                #     ii = np.arange(1,Mx-1)
                #     jj = np.arange(1,My-1)
                #     for iii in ii:
                #         for jjj in jj:
                #             phi = np.pi*np.abs(y2[jjj,iii])/180
                #             dpx1 = (p[jjj,iii] - p[jjj,iii-1])/np.cos(phi)
                #             dpx2 = (p[jjj,iii+1] - p[jjj,iii])/np.cos(phi)
                #             dpy1 = p[jjj,iii] - p[jjj-1,iii]
                #             dpy2 = p[jjj+1,iii] - p[jjj,iii]
                #             dp[jjj,iii] = (dpx1**2 + dpx2**2 )/2+(dpy1**2 + dpy2**2)/2
                #     GrdSlp[:,ttt] = dp.flatten()

                # if slpMemory == True:
                #
                #     with h5py.File(estelaMat, 'r') as f:
                #         # for k in f.keys():
                #         #     print(k)
                #         Xraw = f['full/Xraw'][:]
                #         Y = f['full/Y'][:]
                #         DJF = f['C/traveldays_interp/DJF'][:]
                #         MAM = f['C/traveldays_interp/MAM'][:]
                #         JJA = f['C/traveldays_interp/JJA'][:]
                #         SON = f['C/traveldays_interp/SON'][:]
                #
                #
                #
                #     X_estela = Xraw
                #     tempX = np.where(X_estela < 120)
                #     X_estela[tempX] = X_estela[tempX]+(X_estela[tempX]*0+360)
                #     estelaX = np.hstack((X_estela[:,600:],X_estela[:,0:600]))
                #
                #     Y_estela = Y
                #     estelaY = np.hstack((Y_estela[:,600:],Y_estela[:,0:600]))
                #
                #     #dateTemp = days
                # if mm == 0:
                #     slpMem = slpDownscaled
                #     datesMem = datesAvg
                # else:
                #     slpMem = np.hstack((slpMem, slpDownscaled))
                #     datesMem = np.append(datesMem, datesAvg)
            # slpMem = slpDownscaled
            # datesMem = datesAvg


            if counter == 0:
                SLPS = slpMem
                GRDS = grdMem
                DATES = datesMem
            else:
                SLPS = np.hstack((SLPS, slpMem))
                GRDS = np.hstack((GRDS, grdMem))
                DATES = np.append(DATES, datesMem)

            counter = counter + 1

        # vgrad = np.gradient(SLPS)
        # magGrad = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2)


        print('Extracted until {}-{}'.format(year2, month2))

        from global_land_mask import globe
        wrapLons = np.where((x2 > 180))
        x2[wrapLons] = x2[wrapLons] - 360
        xFlat = x2.flatten()
        yFlat = y2.flatten()

        isOnLandGrid = globe.is_land(y2, x2)
        isOnLandFlat = globe.is_land(yFlat, xFlat)

        x2[wrapLons] = x2[wrapLons] + 360
        xFlat = x2.flatten()


        self.xGrid = x2
        self.yGrid = y2
        self.xFlat = xFlat
        self.yFlat = yFlat
        self.isOnLandGrid = isOnLandGrid
        self.isOnLandFlat = isOnLandFlat
        self.SLPS = SLPS
        self.GRDS = GRDS
        self.DATES = DATES
        self.Mx = Mx
        self.My = My

        import pickle
        samplesPickle = 'slps.pickle'
        outputSamples = {}
        outputSamples['x2'] = x2
        outputSamples['y2'] = y2
        outputSamples['xFlat'] = xFlat
        outputSamples['yFlat'] = yFlat
        outputSamples['isOnLandGrid'] = isOnLandGrid
        outputSamples['isOnLandFlat'] = isOnLandFlat
        outputSamples['SLPS'] = SLPS
        outputSamples['GRDS'] = GRDS
        outputSamples['DATES'] = DATES
        outputSamples['Mx'] = Mx
        outputSamples['My'] = My

        with open(samplesPickle, 'wb') as f:
            pickle.dump(outputSamples, f)

    def pcaOfSlps(self):

        from sklearn.decomposition import PCA

        trimSlps = self.SLPS[~self.isOnLandFlat,:]
        trimGrds = self.GRDS[~self.isOnLandFlat,:]


        SlpGrd = np.hstack((trimSlps.T, trimGrds.T))
        SlpGrdMean = np.mean(SlpGrd, axis=0)
        SlpGrdStd = np.std(SlpGrd, axis=0)
        SlpGrdNorm = (SlpGrd[:, :] - SlpGrdMean) / SlpGrdStd
        SlpGrdNorm[np.isnan(SlpGrdNorm)] = 0

        # principal components analysis
        ipca = PCA(n_components=min(SlpGrdNorm.shape[0], SlpGrdNorm.shape[1]))
        PCs = ipca.fit_transform(SlpGrdNorm)
        EOFs = ipca.components_
        variance = ipca.explained_variance_
        nPercent = variance / np.sum(variance)
        APEV = np.cumsum(variance) / np.sum(variance) * 100.0
        nterm = np.where(APEV <= 0.95 * 100)[0][-1]

        self.SlpGrdMean = SlpGrdMean
        self.SlpGrdStd = SlpGrdStd
        self.SlpGrdNorm = SlpGrdNorm
        self.SlpGrd = SlpGrd
        self.PCs = PCs
        self.EOFs = EOFs
        self.variance = variance
        self.nPercent = nPercent
        self.APEV = APEV
        self.nterm = nterm

        import pickle
        samplesPickle = 'pcas.pickle'
        outputSamples = {}
        outputSamples['SlpGrdMean'] = SlpGrdMean
        outputSamples['SlpGrdStd'] = SlpGrdStd
        outputSamples['SlpGrdNorm'] = SlpGrdNorm
        outputSamples['SlpGrd'] = SlpGrd
        outputSamples['PCs'] = PCs
        outputSamples['EOFs'] = EOFs
        outputSamples['variance'] = variance
        outputSamples['nPercent'] = nPercent
        outputSamples['APEV'] = APEV
        outputSamples['nterm'] = nterm

        with open(samplesPickle, 'wb') as f:
            pickle.dump(outputSamples, f)

    def wtClusters(self,numClusters=49,TCs=True,Basin=b'NA'):
        from functions import sort_cluster_gen_corr_end
        from sklearn.cluster import KMeans
        import numpy as np
        self.numClusters = numClusters
        PCsub = self.PCs[:, :self.nterm + 1]
        EOFsub = self.EOFs[:self.nterm + 1, :]

        if TCs == True:
            print('Downloading the latest IBTRACS data')
            self.basin = Basin
            from urllib import request
            remote_url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc'
            opener = request.build_opener()
            opener.addheaders = [('User-Agent', 'MyApp/1.0')]
            request.install_opener(opener)
            file = 'tcs.nc'
            request.urlretrieve(remote_url, file)
            import xarray as xr
            data = xr.open_dataset(file)
            tcBasin = data['basin']
            TCtime = data['time'].values
            TClon = data['lon'].values
            TClat = data['lat'].values
            TCpres = data['usa_pres']
            TCwind = data['usa_wind']
            import numpy as np

            print('isolating your ocean basin: {}'.format(self.basin))

            indexTC = np.where(tcBasin[:,0]==self.basin)
            tcTime = TCtime[indexTC[0],:]
            tcPres = TCpres[indexTC[0],:]
            tcWind = TCwind[indexTC[0],:]
            tcLon = TClon[indexTC[0],:]
            tcLat = TClat[indexTC[0],:]




            from functions import dt2cal
            tcAllTime = [dt2cal(dt) for dt in tcTime.flatten()]
            tcDailyTime = [np.array([d[0],d[1],d[2]]) for d in tcAllTime]
            # u, idx, counts = np.unique(tcAllTime, axis=0, return_index=True, return_counts=True)
            allTCtimes = np.unique(tcDailyTime, axis=0, return_index=False, return_counts=False)
            recentTCs = np.where(allTCtimes[:,0] > 1940)
            allTCtimes = allTCtimes[recentTCs[0],:]

            import datetime
            def dateDay2datetime(d_vec):
                '''
                Returns datetime list from a datevec matrix
                d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
                '''
                return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]

            def dateDay2datetimeDate(d_vec):
                '''
                Returns datetime list from a datevec matrix
                d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
                '''
                return [datetime.date(d[0], d[1], d[2]) for d in d_vec]

            def datetime2datetimeDate(d_vec):
                '''
                Returns datetime list from a datevec matrix
                d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
                '''
                return [datetime.date(d.year, d.month, d.day) for d in d_vec]

            allTCtimes = np.asarray(dateDay2datetime(allTCtimes))
            import pandas as pd
            df = pd.DataFrame(allTCtimes, columns=['date'])
            dropDups = df.drop_duplicates('date')

            tcDates = dropDups['date'].dt.date.tolist()
            slpDates = datetime2datetimeDate(self.DATES)
            overlap = [x for x in slpDates if x in tcDates]

            ind_dict = dict((k, i) for i, k in enumerate(slpDates))
            inter = set(slpDates).intersection(tcDates)
            indices = [ind_dict[x] for x in inter]
            indices.sort()

            mask = np.ones(len(self.PCs), bool)
            mask[indices] = 0
            pcLess = PCsub[mask,:]

            mask2 = np.zeros(len(self.PCs), np.bool)
            mask2[indices] = 1
            pcTCs = PCsub[mask2,:]

            print('clustering Extra Tropical Days')
            self.num_clustersETC = 49
            kma = KMeans(n_clusters=self.num_clustersETC, n_init=2000).fit(pcLess)
            # groupsize
            _, group_sizeETC = np.unique(kma.labels_, return_counts=True)

            # groups
            d_groupsETC = {}
            for k in range(self.num_clustersETC):
                d_groupsETC['{0}'.format(k)] = np.where(kma.labels_ == k)

            self.groupSizeETC = group_sizeETC
            self.dGroupsETC = d_groupsETC
            # centroids
            self.centroidsETC = np.dot(kma.cluster_centers_, EOFsub)
            # km, x and var_centers
            self.kmETC = np.multiply(
                self.centroidsETC,
                np.tile(self.SlpGrdStd, (self.num_clustersETC, 1))
            ) + np.tile(self.SlpGrdMean, (self.num_clustersETC, 1))
            # sort kmeans
            kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, numClusters)
            self.kmaOrderETC = kma_order
            bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
            for i in range(self.num_clustersETC):
                posc = np.where(kma.labels_ == kma_order[i])
                bmus_corrected[posc] = i
            self.bmus_correctedETC = bmus_corrected
            # reorder centroids
            self.sorted_cenEOFsETC = kma.cluster_centers_[kma_order, :]
            self.sorted_centroidsETC = self.centroidsETC[kma_order, :]

            repmatStd = np.tile(self.SlpGrdStd, (self.num_clustersETC, 1))
            repmatMean = np.tile(self.SlpGrdMean, (self.num_clustersETC, 1))
            self.Km_ETC = np.multiply(self.sorted_centroidsETC, repmatStd) + repmatMean

            print('clustering Tropical Cyclone Days')

            self.num_clustersTC = 21
            kma = KMeans(n_clusters=self.num_clustersTC, n_init=2000).fit(pcTCs)
            # groupsize
            _, group_sizeTC = np.unique(kma.labels_, return_counts=True)
            # groups
            d_groupsTC = {}
            for k in range(self.num_clustersTC):
                d_groupsTC['{0}'.format(k)] = np.where(kma.labels_ == k)
            self.groupSizeTC = group_sizeTC
            self.dGroupsTC = d_groupsTC
            # centroids
            self.centroidsTC = np.dot(kma.cluster_centers_, EOFsub)
            # km, x and var_centers
            self.kmTC = np.multiply(
                self.centroidsTC,
                np.tile(self.SlpGrdStd, (self.num_clustersTC, 1))
            ) + np.tile(self.SlpGrdMean, (self.num_clustersTC, 1))
            self.bmusTC = kma.labels_+49
            repmatStd = np.tile(self.SlpGrdStd, (self.num_clustersTC, 1))
            repmatMean = np.tile(self.SlpGrdMean, (self.num_clustersTC, 1))
            self.Km_TC = np.multiply(self.centroidsTC, repmatStd) + repmatMean
            self.cenEOFsTC = kma.cluster_centers_


            self.bmus = np.nan * np.ones((len(self.PCs)))
            self.bmus[mask] = self.bmus_correctedETC
            self.bmus[mask2] = self.bmusTC
            self.bmus_corrected = self.bmus

            import pickle
            samplesPickle = 'dwts.pickle'
            outputSamples = {}
            outputSamples['bmus'] = self.bmus
            outputSamples['Km_ETC'] = self.Km_ETC
            outputSamples['sorted_centroidsETC'] = self.sorted_centroidsETC
            outputSamples['sorted_cenEOFsETC'] = self.sorted_cenEOFsETC
            outputSamples['bmus_correctedETC'] = self.bmus_correctedETC
            outputSamples['kmaOrderETC'] = self.kmaOrderETC
            outputSamples['dGroupsETC'] = self.dGroupsETC
            outputSamples['groupSizeETC'] = self.groupSizeETC
            # outputSamples['numClustersETC'] = self.numClusters
            outputSamples['numClustersETC'] = self.num_clustersETC
            outputSamples['Km_TC'] = self.Km_TC
            outputSamples['sorted_centroidsTC'] = self.centroidsTC
            outputSamples['sorted_cenEOFsTC'] = self.cenEOFsTC
            outputSamples['bmus_correctedTC'] = self.bmusTC
            # outputSamples['kmaOrderTC'] = self.kmaOrderTC
            outputSamples['dGroupsTC'] = self.dGroupsTC
            outputSamples['groupSizeTC'] = self.groupSizeTC
            outputSamples['numClustersTC'] = self.num_clustersTC
            outputSamples['bmus_corrected'] = self.bmus_corrected

            with open(samplesPickle, 'wb') as f:
                pickle.dump(outputSamples, f)




        else:
            kma = KMeans(n_clusters=numClusters, n_init=2000).fit(PCsub)
            # groupsize
            _, group_sizeETC = np.unique(kma.labels_, return_counts=True)
            # groups
            d_groupsETC = {}
            for k in range(numClusters):
                d_groupsETC['{0}'.format(k)] = np.where(kma.labels_ == k)
            self.groupSizeETC = group_sizeETC
            self.dGroupsETC = d_groupsETC
            # centroids
            centroids = np.dot(kma.cluster_centers_, EOFsub)
            # km, x and var_centers
            km = np.multiply(
                centroids,
                np.tile(self.SlpGrdStd, (numClusters, 1))
            ) + np.tile(self.SlpGrdMean, (numClusters, 1))
            # sort kmeans
            kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, numClusters)
            self.kmaOrderETC = kma_order
            bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
            for i in range(numClusters):
                posc = np.where(kma.labels_ == kma_order[i])
                bmus_corrected[posc] = i
            self.bmus_correctedETC = bmus_corrected
            # reorder centroids
            self.sorted_cenEOFsETC = kma.cluster_centers_[kma_order, :]
            self.sorted_centroidsETC = centroids[kma_order, :]

            repmatStd = np.tile(self.SlpGrdStd, (numClusters, 1))
            repmatMean = np.tile(self.SlpGrdMean, (numClusters, 1))
            self.Km_ETC = np.multiply(self.sorted_centroidsETC, repmatStd) + repmatMean

            import pickle
            samplesPickle = 'dwts.pickle'
            outputSamples = {}
            outputSamples['Km_ETC'] = self.Km_ETC
            outputSamples['sorted_centroidsETC'] = self.sorted_centroidsETC
            outputSamples['sorted_cenEOFsETC'] = self.sorted_cenEOFsETC
            outputSamples['bmus_correctedETC'] = self.bmus_correctedETC
            outputSamples['kmaOrderETC'] = self.kmaOrderETC
            outputSamples['dGroupsETC'] = self.dGroupsETC
            outputSamples['groupSizeETC'] = self.groupSizeETC
            outputSamples['numClustersETC'] = self.numClusters

            with open(samplesPickle, 'wb') as f:
                pickle.dump(outputSamples, f)



    def alrSimulations(self,climate,historicalSimNum,futureSimNum,loadPrevious=False,plotOutput=False):
        from functions import xds_reindex_daily as xr_daily
        from functions import xds_common_dates_daily as xcd_daily

        # xds_MJO_fit = xr.Dataset(
        #     {
        #         'rmm1': (('time',), mjoRmm1),
        #         'rmm2': (('time',), mjoRmm2),
        #     },
        #     coords={'time': [datetime.datetime(mjoYear[r], mjoMonth[r], mjoDay[r]) for r in range(len(mjoDay))]}
        # )
        # # reindex to daily data after 1979-01-01 (avoid NaN)
        # xds_MJO_fit = xr_daily(xds_MJO_fit, datetime.datetime(1979, 6, 1))

        from datetime import datetime, timedelta
        self.xds_KMA_fit = xr.Dataset(
            {
                'bmus': (('time',), self.bmus_corrected),
            },
            coords={'time': self.DATES}
        )


        # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
        self.xds_PCs_fit = xr.Dataset(
            {
                'PC1': (('time',), climate.dailyPC1),
                'PC2': (('time',), climate.dailyPC3),
                'PC3': (('time',), climate.dailyPC3),
            },
            coords={'time': [datetime(climate.mjoYear[r], climate.mjoMonth[r], climate.mjoDay[r]) for r in range(len(climate.mjoDay))]}
        )
        # reindex annual data to daily data
        self.xds_PCs_fit = xr_daily(self.xds_PCs_fit)

        # MJO: RMM1s (Generated with copula simulation. Annual data, parse to daily)
        xds_MJO_fit = xr.Dataset(
            {
                'rmm1': (('time',), climate.mjoRmm1),
                'rmm2': (('time',), climate.mjoRmm2),
            },
            coords={'time': [datetime(climate.mjoYear[r], climate.mjoMonth[r], climate.mjoDay[r]) for r in range(len(climate.mjoDay))]}
            # coords = {'time': timeMJO}
        )
        # reindex to daily data after 1979-01-01 (avoid NaN)
        # xds_MJO_fit = xr_daily(xds_MJO_fit, datetime(1979, 6, 1), datetime(2023, 5, 31))

        # --------------------------------------
        # Mount covariates matrix

        # available data:
        # model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
        # model sim: xds_MJO_sim, xds_PCs_sim

        # covariates: FIT
        # d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_KMA_fit])
        d_covars_fit = xcd_daily([self.xds_PCs_fit, self.xds_KMA_fit])

        # PCs covar
        cov_PCs = self.xds_PCs_fit.sel(time=slice(d_covars_fit[0], d_covars_fit[-1]))
        cov_1 = cov_PCs.PC1.values.reshape(-1, 1)
        cov_2 = cov_PCs.PC2.values.reshape(-1, 1)
        cov_3 = cov_PCs.PC3.values.reshape(-1, 1)

        # MJO covars
        cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0], d_covars_fit[-1]))
        cov_4 = cov_MJO.rmm1.values.reshape(-1, 1)
        cov_5 = cov_MJO.rmm2.values.reshape(-1, 1)

        # join covars and norm.
        cov_T = np.hstack((cov_1, cov_2, cov_3))

        # generate xarray.Dataset
        cov_names = ['PC1', 'PC2', 'PC3']
        self.xds_cov_fit = xr.Dataset(
            {
                'cov_values': (('time', 'cov_names'), cov_T),
            },
            coords={
                'time': d_covars_fit,
                'cov_names': cov_names,
            }
        )

        # use bmus inside covariate time frame
        self.xds_bmus_fit = self.xds_KMA_fit.sel(
            time=slice(d_covars_fit[0], d_covars_fit[-1])
        )

        bmus = self.xds_bmus_fit.bmus

        # Autoregressive logistic wrapper
        num_clusters = 25
        sim_num = 100
        fit_and_save = True  # False for loading
        p_test_ALR = '/Users/dylananderson/Documents/duneLifeCycles/'

        # ALR terms
        self.d_terms_settings = {
            'mk_order': 2,
            'constant': True,
            'long_term': False,
            'seasonality': (False,),  # [2, 4, 6]),
            'covariates': (True, self.xds_cov_fit),
        }
        # Autoregressive logistic wrapper
        ALRW = ALR_WRP(p_test_ALR)
        ALRW.SetFitData(
            num_clusters, self.xds_bmus_fit, self.d_terms_settings)

        ALRW.FitModel(max_iter=20000)

        # p_report = op.join(p_data, 'r_{0}'.format(name_test))

        ALRW.Report_Fit()  # '/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)

        if historicalSimNum > 0:
            # Historical Simulation
            # start simulation at PCs available data
            d1 = datetime(1979, 6, 1)  # x2d(xds_cov_fit.time[0])
            d2 = datetime(2023, 6, 1)  # datetime(d1.year+sim_years, d1.month, d1.day)
            dates_sim = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]
            # print some info
            # print('ALR model fit   : {0} --- {1}'.format(
            #    d_covars_fit[0], d_covars_fit[-1]))
            print('ALR model sim   : {0} --- {1}'.format(
                dates_sim[0], dates_sim[-1]))

            #  launch simulation
            xds_ALR = ALRW.Simulate(
                historicalSimNum, dates_sim[0:-1], self.xds_cov_fit)

            self.historicalDatesSim = dates_sim

            # Save results for matlab plot
            self.historicalBmusSim = xds_ALR.evbmus_sims.values
            # evbmus_probcum = xds_ALR.evbmus_probcum.values

            # convert synthetic markovs to PC values
            # Fill in the Markov chain bmus with RMM vales
            self.rmm1Sims = list()
            self.rmm2Sims = list()
            for kk in range(historicalSimNum):
                tempSimulation = self.historicalBmusSim[:, kk]
                tempPC1 = np.nan * np.ones((np.shape(tempSimulation)))
                tempPC2 = np.nan * np.ones((np.shape(tempSimulation)))

                self.groups = [list(j) for i, j in groupby(tempSimulation)]
                c = 0
                for gg in range(len(self.groups)):
                    getInds = rm.sample(range(1, 100000), len(self.groups[gg]))
                    tempPC1s = self.gevCopulaSims[int(self.groups[gg][0]) - 1][getInds[0], 0]
                    tempPC2s = self.gevCopulaSims[int(self.groups[gg][0]) - 1][getInds[0], 1]
                    tempPC1[c:c + len(self.groups[gg])] = tempPC1s
                    tempPC2[c:c + len(self.groups[gg])] = tempPC2s
                    c = c + len(self.groups[gg])
                self.rmm1Sims.append(tempPC1)
                self.rmm2Sims.append(tempPC2)
            self.mjoHistoricalSimRmm1 = self.rmm1Sims
            self.mjoHistoricalSimRmm2 = self.rmm2Sims

        if futureSimNum > 0:
            futureSims = []
            for simIndex in range(futureSimNum):
                # ALR FUTURE model simulations
                sim_years = 100
                # start simulation at PCs available data
                d1 = datetime(2023, 6, 1)  # x2d(xds_cov_fit.time[0])
                d2 = datetime(2123, 6, 1)  # datetime(d1.year+sim_years, d1.month, d1.day)
                self.future_dates_sim = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]

                d1 = datetime(2023, 6, 1)
                dt = datetime(2023, 6, 1)
                end = datetime(2123, 6, 1)
                step = relativedelta(years=1)
                simAnnualTime = []
                while dt < end:
                    simAnnualTime.append(dt)
                    dt += step

                d1 = datetime(2023, 6, 1)
                dt = datetime(2023, 6, 1)
                end = datetime(2123, 6, 2)
                # step = datetime.timedelta(months=1)
                step = relativedelta(days=1)
                simDailyTime = []
                while dt < end:
                    simDailyTime.append(dt)
                    dt += step
                simDailyDatesMatrix = np.array([[r.year, r.month, r.day] for r in simDailyTime])

                trainingDates = [datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix]
                dailyAWTsim = np.ones((len(trainingDates),))
                dailyPC1sim = np.ones((len(trainingDates),))
                dailyPC2sim = np.ones((len(trainingDates),))
                dailyPC3sim = np.ones((len(trainingDates),))

                awtBMUsim = self.awtBmusSim[simIndex][0:100]  # [0:len(awt_bmus)]
                awtPC1sim = self.pc1Sims[simIndex][0:100]  # [0:len(awt_bmus)]
                awtPC2sim = self.pc2Sims[simIndex][0:100]  # [0:len(awt_bmus)]
                awtPC3sim = self.pc3Sims[simIndex][0:100]  # [0:len(awt_bmus)]
                dailyDatesSWTyear = np.array([r[0] for r in simDailyDatesMatrix])
                dailyDatesSWTmonth = np.array([r[1] for r in simDailyDatesMatrix])
                dailyDatesSWTday = np.array([r[2] for r in simDailyDatesMatrix])
                normPC1 = awtPC1sim
                normPC2 = awtPC2sim
                normPC3 = awtPC3sim

                for i in range(len(awtBMUsim)):
                    sSeason = np.where((simDailyDatesMatrix[:, 0] == simAnnualTime[i].year) & (
                            simDailyDatesMatrix[:, 1] == simAnnualTime[i].month) & (simDailyDatesMatrix[:, 2] == 1))
                    ssSeason = np.where((simDailyDatesMatrix[:, 0] == simAnnualTime[i].year + 1) & (
                            simDailyDatesMatrix[:, 1] == simAnnualTime[i].month) & (simDailyDatesMatrix[:, 2] == 1))

                    dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1] = awtBMUsim[i] * dailyAWTsim[
                                                                                   sSeason[0][0]:ssSeason[0][0] + 1]
                    dailyPC1sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC1[i] * np.ones(
                        len(dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1]), )
                    dailyPC2sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC2[i] * np.ones(
                        len(dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1]), )
                    dailyPC3sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC3[i] * np.ones(
                        len(dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1]), )

                # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
                self.xds_PCs_sim = xr.Dataset(
                    {
                        'PC1': (('time',), dailyPC1sim),
                        'PC2': (('time',), dailyPC2sim),
                        'PC3': (('time',), dailyPC3sim),
                    },
                    coords={'time': [datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix]}
                )
                # reindex annual data to daily data
                self.xds_PCs_sim = xr_daily(self.xds_PCs_sim)

                d_covars_sim = xcd_daily([self.xds_PCs_sim])

                # PCs covar
                cov_PCs = self.xds_PCs_sim.sel(time=slice(d_covars_sim[0], d_covars_sim[-1]))
                cov_1 = cov_PCs.PC1.values.reshape(-1, 1)
                cov_2 = cov_PCs.PC2.values.reshape(-1, 1)
                cov_3 = cov_PCs.PC3.values.reshape(-1, 1)

                # MJO covars
                # cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0], d_covars_fit[-1]))
                # cov_4 = cov_MJO.rmm1.values.reshape(-1, 1)
                # cov_5 = cov_MJO.rmm2.values.reshape(-1, 1)

                # join covars and norm.
                cov_T = np.hstack((cov_1, cov_2, cov_3))

                # generate xarray.Dataset
                cov_names = ['PC1', 'PC2', 'PC3']
                self.xds_cov_sim = xr.Dataset(
                    {
                        'cov_values': (('time', 'cov_names'), cov_T),
                    },
                    coords={
                        'time': d_covars_sim,
                        'cov_names': cov_names,
                    }
                )

                #  launch simulation
                xds_ALRfuture = ALRW.Simulate(
                    1, self.future_dates_sim, self.xds_cov_sim)

                self.futureDatesSim = self.future_dates_sim

                futureSims.append(xds_ALRfuture.evbmus_sims.values)

            # Save results for matlab plot
            self.futureBmusSim = futureSims
            # evbmus_probcum = xds_ALR.evbmus_probcum.values
            # convert synthetic markovs to PC values
            # Fill in the Markov chain bmus with RMM vales
            rmm1Sims = list()
            rmm2Sims = list()
            for kk in range(len(self.futureBmusSim)):
                tempSimulation = self.futureBmusSim[kk]
                tempPC1 = np.nan * np.ones((np.shape(tempSimulation)))
                tempPC2 = np.nan * np.ones((np.shape(tempSimulation)))

                groups = [list(j) for i, j in groupby(tempSimulation)]
                c = 0
                for gg in range(len(groups)):
                    getInds = rm.sample(range(1, 100000), len(groups[gg]))
                    tempPC1s = self.gevCopulaSims[int(groups[gg][0] - 1)][getInds[0], 0]
                    tempPC2s = self.gevCopulaSims[int(groups[gg][0] - 1)][getInds[0], 1]
                    tempPC1[c:c + len(groups[gg])] = tempPC1s
                    tempPC2[c:c + len(groups[gg])] = tempPC2s
                    c = c + len(groups[gg])
                rmm1Sims.append(tempPC1)
                rmm2Sims.append(tempPC2)
            self.mjoFutureSimRmm1 = rmm1Sims
            self.mjoFutureSimRmm2 = rmm2Sims


    def separateHistoricalHydrographs(self,metOcean,numRealizations=100,shoreNorm=90):
        import itertools
        import operator
        from datetime import timedelta
        import random
        import numpy as np

        bmus = self.bmus
        time = self.DATES

        dt = time[0]
        end = time[-1]
        step = timedelta(days=1)
        midnightTime = []
        while dt <= end:
            midnightTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        groupedList = list()
        groupLengthList = list()
        bmuGroupList = list()
        timeGroupList = list()
        # for hh in range(1):
        #     print('breaking up hydrogrpahs for historical {}'.format(hh))
        #     # bmus = evbmus_sim[:,hh]
        tempBmusGroup = [[e[0] for e in d[1]] for d in
                             itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]
        groupedList.append(tempBmusGroup)
        groupLengthList.append(np.asarray([len(i) for i in tempBmusGroup]))
        bmuGroupList.append(np.asarray([bmus[i[0]] for i in tempBmusGroup]))
        timeGroupList.append([np.asarray(midnightTime)[i] for i in tempBmusGroup])

        simBmuChopped = []
        simBmuLengthChopped = []
        simBmuGroupsChopped = []
        for pp in range(numRealizations):

            print('working on realization #{}'.format(pp))
            bmuGroup = bmuGroupList[0]
            groupLength = groupLengthList[0]
            grouped = groupedList[0]
            simGroupLength = []
            simGrouped = []
            simBmu = []
            for i in range(len(groupLength)):
                # if np.remainder(i,10000) == 0:
                #     print('done with {} hydrographs'.format(i))
                tempGrouped = grouped[i]
                tempBmu = int(bmuGroup[i])
                remainingDays = groupLength[i] - 5
                if groupLength[i] < 5:
                    simGroupLength.append(int(groupLength[i]))
                    simGrouped.append(grouped[i])
                    simBmu.append(tempBmu)
                else:
                    counter = 0
                    while (len(grouped[i]) - counter) > 5:
                        # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                        # random days between 3 and 5
                        randLength = random.randint(1, 3) + 2
                        # add this to the record
                        simGroupLength.append(int(randLength))
                        # simGrouped.append(tempGrouped[0:randLength])
                        # print('should be adding {}'.format(grouped[i][counter:counter+randLength]))
                        simGrouped.append(grouped[i][counter:counter + randLength])
                        simBmu.append(tempBmu)
                        # remove those from the next step
                        # tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))
                        # do we continue forward
                        remainingDays = remainingDays - randLength
                        counter = counter + randLength

                    if (len(grouped[i]) - counter) > 0:
                        simGroupLength.append(int((len(grouped[i]) - counter)))
                        # simGrouped.append(tempGrouped[0:])
                        simGrouped.append(grouped[i][counter:])
                        simBmu.append(tempBmu)
            simBmuLengthChopped.append(np.asarray(simGroupLength))
            simBmuGroupsChopped.append(simGrouped)
            simBmuChopped.append(np.asarray(simBmu))

        self.histBmuLengthCopped = simBmuLengthChopped
        self.histBmuGroupsChopped = simBmuGroupsChopped
        self.histBmuChopped = simBmuChopped

        from dateutil.relativedelta import relativedelta
        from datetime import datetime
        st = time[0]
        end = datetime(self.endTime[0], self.endTime[1] + 1, 1)
        step = relativedelta(hours=1)
        hourTime = []
        while st < end:
            hourTime.append(st)  # .strftime('%Y-%m-%d'))
            st += step

        beginTime = np.where(
            (np.asarray(hourTime) == datetime(time[0].year, time[0].month, time[0].day, 0, 0)))
        endingTime = np.where(
            (np.asarray(hourTime) == datetime(time[-1].year, time[-1].month, time[-1].day, 0, 0)))

        wh = metOcean.Hs#[beginTime[0][0]:endingTime[0][0] + 24]
        tp = metOcean.Tp#[beginTime[0][0]:endingTime[0][0] + 24]
        dm = metOcean.Dm#[beginTime[0][0]:endingTime[0][0] + 24]
        ntr = metOcean.resWL#[beginTime[0][0]:endingTime[0][0] + 24]

        waveNorm = dm - shoreNorm
        neg = np.where((waveNorm > 180))
        waveNorm[neg[0]] = waveNorm[neg[0]] - 360
        neg2 = np.where((waveNorm < -180))
        waveNorm[neg2[0]] = waveNorm[neg2[0]] + 360
        dmOG = dm
        dm = waveNorm

        startTimes = [i[0] for i in timeGroupList[0]]
        endTimes = [i[-1] for i in timeGroupList[0]]
        time_all = metOcean.timeWave
        tNTR = metOcean.timeWL

        hydrosInds = np.unique(bmuGroup)

        hydros = list()
        c = 0
        for p in range(len(np.unique(bmuGroup))):

            tempInd = p
            # print('working on bmu = {}'.format(tempInd))
            index = np.where((bmuGroup == tempInd))[0][:]

            # print('should have at least {} hydrographs in it'.format(len(index)))
            subLength = groupLength[index]
            m = np.ceil(np.sqrt(len(index)))
            tempList = list()
            counter = 0
            for i in range(len(index)):
                st = startTimes[index[i]]
                et = endTimes[index[i]] + timedelta(days=1)

                waveInd = np.where((time_all < et) & (time_all >= st))
                ntrInd = np.where((tNTR < et) & (tNTR >= st))

                if len(waveInd[0]) > 0 and len(ntrInd[0]):
                    newTime = startTimes[index[i]]

                    tempHydroLength = subLength[i]

                    waveInd = np.where((time_all < et) & (time_all >= newTime))
                    ntrInd = np.where((tNTR < et) & (tNTR >= newTime))

                    counter = counter + 1

                    tempDict = dict()
                    tempDict['time'] = time_all[waveInd[0]]
                    tempDict['numDays'] = subLength[i]
                    tempDict['hs'] = wh[waveInd[0]]
                    tempDict['tp'] = tp[waveInd[0]]
                    tempDict['dm'] = dm[waveInd[0]]
                    tempDict['ntr'] = ntr[ntrInd[0]]
                    tempDict['cop'] = np.asarray([np.nanmin(wh[waveInd[0]]), np.nanmax(wh[waveInd[0]]),
                                                np.nanmin(tp[waveInd[0]]), np.nanmax(tp[waveInd[0]]),
                                                np.nanmean(dm[waveInd[0]]),np.nanmean(ntr[ntrInd[0]])])

                    tempDict['hsMin'] = np.nanmin(wh[waveInd[0]])
                    tempDict['hsMax'] = np.nanmax(wh[waveInd[0]])
                    tempDict['tpMin'] = np.nanmin(tp[waveInd[0]])
                    tempDict['tpMax'] = np.nanmax(tp[waveInd[0]])
                    tempDict['dmMean'] = np.nanmean(dm[waveInd[0]])
                    tempDict['ntrMean'] = np.nanmean(ntr[ntrInd[0]])
                    tempDict['ntrMin'] = np.nanmin(ntr[ntrInd[0]])
                    tempDict['ntrMax'] = np.nanmax(ntr[ntrInd[0]])
                    tempList.append(tempDict)
                    tempList.append(tempDict)
            print('we collected {} of {} hydrographs due to WL data gaps in weather pattern {}'.format(counter, len(index), p))
            hydros.append(tempList)
        self.hydros = hydros
        self.bmuGroup = bmuGroupList[0]
        self.groupLength = groupLengthList[0]
        self.grouped = groupedList[0]
            #
            #         while tempHydroLength > 1:
            #             # randLength = random.randint(1, 2)
            #             etNew = newTime + timedelta(days=1)
            #             # if etNew >= et:
            #             #     etNew = newTime + timedelta(days=1)
            #             waveInd = np.where((time_all < etNew) & (time_all >= newTime))
            #             # fetchInd = np.where((np.asarray(timeArrayIce) == newTime))
            #             ntrInd = np.where((tNTR < etNew) & (tNTR >= newTime))
            #
            #             if len(waveInd[0]) > 0 and len(ntrInd[0]):
            #                 deltaDays = et - etNew
            #                 c = c + 1
            #                 counter = counter + 1
            #                 # if counter > 15:
            #                 #     print('we have split 15 times')
            #
            #                 tempDict = dict()
            #                 tempDict['time'] = time_all[waveInd[0]]
            #                 tempDict['numDays'] = subLength[i]
            #                 tempDict['hs'] = wh[waveInd[0]]
            #                 tempDict['tp'] = tp[waveInd[0]]
            #                 tempDict['dm'] = dm[waveInd[0]]
            #                 # tempDict['v10'] = v10[waveInd[0]]
            #                 # tempDict['u10'] = u10[waveInd[0]]
            #                 # tempDict['sst'] = sst[waveInd[0]]
            #                 # tempDict['ssr'] = ssr[waveInd[0]]
            #                 # tempDict['t2m'] = t2m[waveInd[0]]
            #                 tempDict['ntr'] = ntr[ntrInd[0]]
            #
            #                 # tempDict['fetch'] = areaBelow[fetchInd[0][0]]
            #                 tempDict['cop'] = np.asarray([np.nanmin(wh[waveInd[0]]), np.nanmax(wh[waveInd[0]]),
            #                                               np.nanmin(tp[waveInd[0]]), np.nanmax(tp[waveInd[0]]),
            #                                               np.nanmean(dm[waveInd[0]]),
            #                                               # np.nanmean(u10[waveInd[0]]),
            #                                               # np.nanmean(v10[waveInd[0]]), np.nanmean(sst[waveInd[0]]),
            #                                               # np.nanmean(ssr[waveInd[0]]), np.nanmean(t2m[waveInd[0]]),
            #                                               # areaBelow[fetchInd[0][0]],
            #                                               np.nanmean(ntr[ntrInd[0]])])
            #
            #                 tempDict['hsMin'] = np.nanmin(wh[waveInd[0]])
            #                 tempDict['hsMax'] = np.nanmax(wh[waveInd[0]])
            #                 tempDict['tpMin'] = np.nanmin(tp[waveInd[0]])
            #                 tempDict['tpMax'] = np.nanmax(tp[waveInd[0]])
            #                 tempDict['dmMean'] = np.nanmean(dm[waveInd[0]])
            #                 # tempDict['u10Mean'] = np.nanmean(u10[waveInd[0]])
            #                 # tempDict['u10Max'] = np.nanmax(u10[waveInd[0]])
            #                 # tempDict['u10Min'] = np.nanmin(u10[waveInd[0]])
            #                 # tempDict['v10Max'] = np.nanmax(v10[waveInd[0]])
            #                 # tempDict['v10Mean'] = np.nanmean(v10[waveInd[0]])
            #                 # tempDict['v10Min'] = np.nanmin(v10[waveInd[0]])
            #                 # tempDict['sstMean'] = np.nanmean(sst[waveInd[0]])
            #                 # tempDict['ssrMean'] = np.nanmean(ssr[waveInd[0]])
            #                 # tempDict['ssrMin'] = np.nanmin(ssr[waveInd[0]])
            #                 # tempDict['ssrMax'] = np.nanmax(ssr[waveInd[0]])
            #                 # tempDict['t2mMean'] = np.nanmean(t2m[waveInd[0]])
            #                 # tempDict['t2mMin'] = np.nanmin(t2m[waveInd[0]])
            #                 # tempDict['t2mMax'] = np.nanmax(t2m[waveInd[0]])
            #                 tempDict['ntrMean'] = np.nanmean(ntr[ntrInd[0]])
            #                 tempDict['ntrMin'] = np.nanmin(ntr[ntrInd[0]])
            #                 tempDict['ntrMax'] = np.nanmax(ntr[ntrInd[0]])
            #                 tempList.append(tempDict)
            #             tempHydroLength = tempHydroLength - 1
            #             newTime = etNew
            #         else:
            #             if len(waveInd[0]) > 0 and len(ntrInd[0]):
            #                 waveInd = np.where((time_all < et) & (time_all >= newTime))
            #                 # fetchInd = np.where((np.asarray(timeArrayIce) == newTime))
            #                 ntrInd = np.where((tNTR < et) & (tNTR >= newTime))
            #
            #                 c = c + 1
            #                 counter = counter + 1
            #                 tempDict = dict()
            #                 tempDict['time'] = time_all[waveInd[0]]
            #                 tempDict['numDays'] = subLength[i]
            #                 tempDict['hs'] = wh[waveInd[0]]
            #                 tempDict['tp'] = tp[waveInd[0]]
            #                 tempDict['dm'] = dm[waveInd[0]]
            #                 # tempDict['v10'] = v10[waveInd[0]]
            #                 # tempDict['u10'] = u10[waveInd[0]]
            #                 # tempDict['sst'] = sst[waveInd[0]]
            #                 # tempDict['ssr'] = ssr[waveInd[0]]
            #                 # tempDict['t2m'] = t2m[waveInd[0]]
            #                 tempDict['ntr'] = ntr[ntrInd[0]]
            #
            #                 # tempDict['fetch'] = areaBelow[fetchInd[0][0]]
            #                 tempDict['cop'] = np.asarray([np.nanmin(wh[waveInd[0]]), np.nanmax(wh[waveInd[0]]),
            #                                               np.nanmin(tp[waveInd[0]]), np.nanmax(tp[waveInd[0]]),
            #                                               np.nanmean(dm[waveInd[0]]),
            #                                               # np.nanmean(u10[waveInd[0]]),
            #                                               # np.nanmean(v10[waveInd[0]]), np.nanmean(sst[waveInd[0]]),
            #                                               # np.nanmean(ssr[waveInd[0]]), np.nanmean(t2m[waveInd[0]]),
            #                                               # areaBelow[fetchInd[0][0]],
            #                                               np.nanmean(ntr[ntrInd[0]])])
            #                 tempDict['hsMin'] = np.nanmin(wh[waveInd[0]])
            #                 tempDict['hsMax'] = np.nanmax(wh[waveInd[0]])
            #                 tempDict['tpMin'] = np.nanmin(tp[waveInd[0]])
            #                 tempDict['tpMax'] = np.nanmax(tp[waveInd[0]])
            #                 tempDict['dmMean'] = np.nanmean(dm[waveInd[0]])
            #                 # tempDict['u10Mean'] = np.nanmean(u10[waveInd[0]])
            #                 # tempDict['u10Max'] = np.nanmax(u10[waveInd[0]])
            #                 # tempDict['u10Min'] = np.nanmin(u10[waveInd[0]])
            #                 # tempDict['v10Max'] = np.nanmax(v10[waveInd[0]])
            #                 # tempDict['v10Mean'] = np.nanmean(v10[waveInd[0]])
            #                 # tempDict['v10Min'] = np.nanmin(v10[waveInd[0]])
            #                 # tempDict['sstMean'] = np.nanmean(sst[waveInd[0]])
            #                 # tempDict['ssrMean'] = np.nanmean(ssr[waveInd[0]])
            #                 # tempDict['ssrMin'] = np.nanmin(ssr[waveInd[0]])
            #                 # tempDict['ssrMax'] = np.nanmax(ssr[waveInd[0]])
            #                 # tempDict['t2mMean'] = np.nanmean(t2m[waveInd[0]])
            #                 # tempDict['t2mMin'] = np.nanmin(t2m[waveInd[0]])
            #                 # tempDict['t2mMax'] = np.nanmax(t2m[waveInd[0]])
            #                 tempDict['ntrMean'] = np.nanmean(ntr[ntrInd[0]])
            #                 tempDict['ntrMin'] = np.nanmin(ntr[ntrInd[0]])
            #                 tempDict['ntrMax'] = np.nanmax(ntr[ntrInd[0]])
            #             tempList.append(tempDict)
            # print('we have split {} times in bmu {}'.format(counter, p))
            # hydros.append(tempList)



    def metOceanCopulas(self):
        import itertools
        import operator
        from datetime import timedelta
        import random
        import numpy as np
        bmus = self.bmus

        ### TODO: make a copula fit for each of the 70 DWTs
        copulaData = list()
        # copulaDataOnlyWaves = list()
        for i in range(len(np.unique(bmus))):
            tempHydros = self.hydros[i]
            dataCop = []
            # dataCopOnlyWaves = []
            for kk in range(len(tempHydros)):
                dataCop.append(list([tempHydros[kk]['hsMax'], tempHydros[kk]['hsMin'], tempHydros[kk]['tpMax'],
                                     tempHydros[kk]['tpMin'], tempHydros[kk]['dmMean'], tempHydros[kk]['ntrMean'], len(tempHydros[kk]['time']), kk]))

                if np.isnan(tempHydros[kk]['hsMax'])==1:
                    print('no waves here')

                # else:
                #     dataCopOnlyWaves.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
                #                      tempHydros[kk]['tpMin'], tempHydros[kk]['dmMean'], tempHydros[kk]['u10Max'],
                #                      tempHydros[kk]['u10Min'], tempHydros[kk]['v10Max'], tempHydros[kk]['v10Min'],
                #                      tempHydros[kk]['ssrMean'], tempHydros[kk]['t2mMean'],
                #                      tempHydros[kk]['fetch'], tempHydros[kk]['ntrMean'], tempHydros[kk]['sstMean'],len(tempHydros[kk]['time']),kk]))

            copulaData.append(dataCop)
            # copulaDataOnlyWaves.append(dataCopOnlyWaves)

        bmuDataNormalized = list()
        bmuDataMin = list()
        bmuDataMax = list()
        bmuDataStd = list()
        for i in range(len(np.unique(bmus))):
            temporCopula = np.asarray(copulaData[i])
            if len(temporCopula) == 0:
                bmuDataNormalized.append(np.vstack((0, 0)).T)
                bmuDataMin.append([0, 0])
                bmuDataMax.append([0, 0])
                bmuDataStd.append([0, 0])
            else:
                dataHs = np.array([sub[0] for sub in copulaData[i]])
                data = temporCopula[~np.isnan(dataHs)]
                data2 = data[~np.isnan(data[:, 0])]
                if len(data2) == 0:
                    print('woah, no waves here bub')
                    bmuDataNormalized.append(np.vstack((0, 0)).T)
                    bmuDataMin.append([0, 0])
                    bmuDataMax.append([0, 0])
                    bmuDataStd.append([0, 0])
                else:

                    maxDm = np.nanmax(data2[:, 4])
                    minDm = np.nanmin(data2[:, 4])
                    stdDm = np.nanstd(data2[:, 4])
                    dmNorm = (data2[:, 4] - minDm) / (maxDm - minDm)
                    maxSs = np.nanmax(data2[:, 5])
                    minSs = np.nanmin(data2[:, 5])
                    stdSs = np.nanstd(data2[:, 5])
                    ssNorm = (data2[:, 5] - minSs) / (maxSs - minSs)
                    bmuDataNormalized.append(np.vstack((dmNorm, ssNorm)).T)
                    bmuDataMin.append([minDm, minSs])
                    bmuDataMax.append([maxDm, maxSs])
                    bmuDataStd.append([stdDm, stdSs])

        normalizedHydros = list()
        for i in range(len(np.unique(bmus))):
            tempHydros = self.hydros[i]
            tempList = list()
            for mm in range(len(tempHydros)):
                if np.isnan(tempHydros[mm]['hsMin']):
                    print('no waves')
                else:
                    tempDict = dict()
                    tempDict['hsNorm'] = (tempHydros[mm]['hs'] - tempHydros[mm]['hsMin']) / (
                                tempHydros[mm]['hsMax'] - tempHydros[mm]['hsMin'])
                    tempDict['tpNorm'] = (tempHydros[mm]['tp'] - tempHydros[mm]['tpMin']) / (
                                tempHydros[mm]['tpMax'] - tempHydros[mm]['tpMin'])
                    tempDict['timeNorm'] = np.arange(0, 1, 1 / len(tempHydros[mm]['time']))[0:len(tempDict['hsNorm'])]
                    tempDict['dmNorm'] = (tempHydros[mm]['dm']) - tempHydros[mm]['dmMean']
                    # tempDict['uNorm'] = (tempHydros[mm]['u10'] - tempHydros[mm]['u10Min']) / (
                    #         tempHydros[mm]['u10Max'] - tempHydros[mm]['u10Min'])
                    # tempDict['vNorm'] = (tempHydros[mm]['v10'] - tempHydros[mm]['v10Min']) / (
                    #         tempHydros[mm]['v10Max'] - tempHydros[mm]['v10Min'])
                    # tempDict['ntrNorm'] = (tempHydros[mm]['ntr'] - tempHydros[mm]['ntrMin']) / (tempHydros[mm]['ntrMax']- tempHydros[mm]['ntrMin'])
                    tempDict['ntrNorm'] = (tempHydros[mm]['ntr'] - tempHydros[mm]['ntrMean'])

                    tempList.append(tempDict)
            normalizedHydros.append(tempList)

        self.normalizedHydros = normalizedHydros
        self.bmuDataMin = bmuDataMin
        self.bmuDataMax = bmuDataMax
        self.bmuDataStd = bmuDataStd
        self.bmuDataNormalized = bmuDataNormalized
        self.copulaData = copulaData

        from functions import copulaSimulation

        gevCopulaSims = list()
        for i in range(len(np.unique(bmus))):
            tempCopula = np.asarray(copulaData[i])
            if len(tempCopula) == 0:
                # Hsmax, Hsmin, Tpmax, Tpmin, Dmmean, u10max, u10min, v10max, v10min, Ssrmean, T2mmean, Fetch, NTRmean, Sstmean, time, kk
                data2 = [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan]]
                data = data2
            else:
                dataHs = np.array([sub[0] for sub in copulaData[i]])
                data = tempCopula[~np.isnan(dataHs)]
                data2 = data[~np.isnan(data[:, 1])]

            print('{} hydrographs of {} in DWT {}'.format(len(data2), len(data), i))

            if len(data2) > 100:
                # # if i == 59 or i == 55 or i == 65 or i == 47 or i == 66 or i == 53 or i == 56:
                # if i == 47 or i == 66 or i == 53 or i == 56:
                #
                #     kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', ]
                # else:
                kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE']
            elif len(data2) == 3 or len(data2) == 2:
                kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE']
                data2 = np.vstack((data2, data2 - data2 * 0.1))
            else:
                kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE']

            if len(data2) <= 1:
                samples5 = np.zeros((100000, 6))
            else:
                samples = copulaSimulation(data2[:, 0:6], kernels, 100000)

                negIndex1 = np.where(samples[:, 0] > 0.1)
                samples2 = samples[negIndex1]

                negIndex2 = np.where(samples2[:, 1] > 0.02)
                samples3 = samples2[negIndex2]

                negIndex3 = np.where(samples3[:, 2] > 1.5)
                samples4 = samples3[negIndex3]
                negIndex4 = np.where(samples4[:, 3] > 0.5)
                samples5 = samples4[negIndex4]

                cutoff = 1.2 * np.nanmax(tempCopula[:, 0])
                toobig = np.where(samples5[:, 0] < cutoff)
                samples5 = samples5[toobig]

            gevCopulaSims.append(samples5)
