
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

    def wtClusters(self,numClusters=49):
        from functions import sort_cluster_gen_corr_end
        from sklearn.cluster import KMeans
        self.numClusters = numClusters
        PCsub = self.PCs[:, :self.nterm + 1]
        EOFsub = self.EOFs[:self.nterm + 1, :]
        kma = KMeans(n_clusters=numClusters, n_init=2000).fit(PCsub)
        # groupsize
        _, group_size = np.unique(kma.labels_, return_counts=True)
        # groups
        d_groups = {}
        for k in range(numClusters):
            d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
        self.groupSize = group_size
        self.dGroups = d_groups
        # centroids
        centroids = np.dot(kma.cluster_centers_, EOFsub)
        # km, x and var_centers
        km = np.multiply(
            centroids,
            np.tile(self.SlpGrdStd, (numClusters, 1))
        ) + np.tile(self.SlpGrdMean, (numClusters, 1))
        # sort kmeans
        kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, numClusters)
        self.kmaOrder = kma_order
        bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
        for i in range(numClusters):
            posc = np.where(kma.labels_ == kma_order[i])
            bmus_corrected[posc] = i
        self.bmus_corrected = bmus_corrected
        # reorder centroids
        self.sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
        self.sorted_centroids = centroids[kma_order, :]

        repmatStd = np.tile(self.SlpGrdStd, (numClusters, 1))
        repmatMean = np.tile(self.SlpGrdMean, (numClusters, 1))
        self.Km_ = np.multiply(self.sorted_centroids, repmatStd) + repmatMean

        import pickle
        samplesPickle = 'dwts.pickle'
        outputSamples = {}
        outputSamples['Km_'] = self.Km_
        outputSamples['sorted_centroids'] = self.sorted_centroids
        outputSamples['sorted_cenEOFs'] = self.sorted_cenEOFs
        outputSamples['bmus_corrected'] = self.bmus_corrected
        outputSamples['kmaOrder'] = self.kmaOrder
        outputSamples['dGroups'] = self.dGroups
        outputSamples['groupSize'] = self.groupSize
        outputSamples['numClusters'] = self.numClusters

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

    #
    # def separateDistributions(self):
    #     import itertools
    #     import operator
    #     from datetime import timedelta
    #
    #     dt = datetime(1979, 2, 1)
    #     end = datetime(2022, 6, 1)
    #     step = timedelta(days=1)
    #     midnightTime = []
    #     while dt < end:
    #         midnightTime.append(dt)  # .strftime('%Y-%m-%d'))
    #         dt += step
    #
    #     bmus = newBmus[0:len(midnightTime)]
    #
    #     grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]
    #
    #     groupLength = np.asarray([len(i) for i in grouped])
    #     bmuGroup = np.asarray([bmus[i[0]] for i in grouped])
    #     timeGroup = [np.asarray(midnightTime)[i] for i in grouped]
    #
    #







