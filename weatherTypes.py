
import numpy as np
import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset
import xarray as xr
from itertools import groupby

class weatherTypes():
    '''
    Class containing camera data and functions'''

    def __init__(self, **kwargs):

        self.lonLeft = kwargs.get('lonLeft', 270)
        self.lonRight = kwargs.get('lonRight',350)
        self.latBot = kwargs.get('latBot', 15)
        self.latTop = kwargs.get('latTop', 55)
        self.resolution = kwargs.get('resolution',1)
        self.resolutionLocal = kwargs.get('resolutionLocal',0.5)

        self.avgTime = kwargs.get('avgTime',24)
        self.startTime = kwargs.get('startTime',[1979,1,1])
        self.endTime = kwargs.get('endTime',[2024,5,31])
        self.slpMemory = kwargs.get('slpMemory',False)
        self.localMemory = kwargs.get('localMemory',False)
        self.slpPath = kwargs.get('slpPath')
        self.minGroupSize = kwargs.get('minGroupSize',50)
        self.savePath = kwargs.get('savePath',os.getcwd())
        self.basin = kwargs.get('basin','atlantic')


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

                # inder = np.where(xFlat < 180)
                # xFlat[inder] = xFlat[inder] + 360

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

    def extractCFSR(self,printToScreen=False,estelaMat=None,loadPrior=False,loadPickle='./'):
        '''
        This function is utilized for opening *.raw Argus files prior to a debayering task,
        and adds the

        Must call this function before calling any debayering functions
        '''

        if loadPrior==True:
            import pickle
            with open(loadPickle, "rb") as input_file:
                slps = pickle.load(input_file)
            self.DATES = slps['DATES']
            self.xGrid = slps['x2']
            self.yGrid = slps['y2']
            self.xFlat = slps['xFlat']
            self.yFlat = slps['yFlat']
            self.isOnLandGrid = slps['isOnLandGrid']
            self.isOnLandFlat = slps['isOnLandFlat']
            self.SLPS = slps['SLPS']
            self.GRDS = slps['GRDS']
            self.Mx = slps['Mx']
            self.My = slps['My']
            print('loaded prior SLP predictor')

        else:


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


            def ceilPartial(value, resolution):
                return np.ceil(value / resolution) * resolution

            def floorPartial(value, resolution):
                return np.floor(value / resolution) * resolution

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

            def roundPartial(value, resolution):
                return round(value / resolution) * resolution


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

                    if printToScreen == True:
                        print('{}-{}'.format(yearExtract, monthExtract))


                    if (yearExtract == 2025 and monthExtract >= 2) or yearExtract > 2025:
                        time = data.variables['time']
                        from datetime import timedelta
                        from calendar import monthrange
                        import zoneinfo  # Python ≥3.9; otherwise use `pytz`
                        #def hourly_vector_stdlib(year: int, month: int, tz: str | None = None):
                        hours_in_month = monthrange(yearExtract, monthExtract)[1] * 24
                        #tzinfo = zoneinfo.ZoneInfo(tz) if tz else None
                        start = datetime(yearExtract, monthExtract, 1, 0)#, tzinfo=tzinfo)
                        dates = [start + timedelta(hours=h) for h in range(hours_in_month)]

                        # Extracting SLP fields, need to account for wrap around international date line
                        if self.lonLeft > self.lonRight:
                            # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                            slp1 = data.variables['Pressure_reduced_to_MSL_msl'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                            slp2 = data.variables['Pressure_reduced_to_MSL_msl'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                   0:(pos_lon2[0][0] + 1)]
                            slp = np.concatenate((slp1, slp2), axis=2)
                        else:
                            slp = data.variables['Pressure_reduced_to_MSL_msl'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                  (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]
                    else:
                        time = data.variables['valid_date_time']
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


                        if self.basin == 'atlantic':
                            inder = np.where(xFlat < 180)
                            xFlat[inder] = xFlat[inder] + 360


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

                    # print('Extracted until {}-{}'.format(year2, month2))

                    from global_land_mask import globe
                    wrapLons = np.where((x2 > 180))
                    x2[wrapLons] = x2[wrapLons] - 360
                    xFlat = x2.flatten()
                    yFlat = y2.flatten()

                    isOnLandGrid = globe.is_land(y2, x2)
                    isOnLandFlat = globe.is_land(yFlat, xFlat)

                    x2[wrapLons] = x2[wrapLons] + 360
                    xFlat = x2.flatten()

                    trimSlps = slpMem[~isOnLandFlat, :]
                    trimGrds = grdMem[~isOnLandFlat, :]

                    if counter == 0:
                        SLPS = trimSlps
                        GRDS = trimGrds
                        DATES = datesMem
                    else:
                        SLPS = np.hstack((SLPS, trimSlps))
                        GRDS = np.hstack((GRDS, trimGrds))
                        DATES = np.append(DATES, datesMem)

                    counter = counter + 1

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
                        print('{}-{}'.format(yearExtract, monthExtract))

                        if (yearExtract == 2025 and monthExtract >= 1) or yearExtract > 2025:
                            time = data.variables['time']
                            from datetime import timedelta
                            from calendar import monthrange
                            import zoneinfo  # Python ≥3.9; otherwise use `pytz`
                            # def hourly_vector_stdlib(year: int, month: int, tz: str | None = None):
                            hours_in_month = monthrange(yearExtract, monthExtract)[1] * 24
                            # tzinfo = zoneinfo.ZoneInfo(tz) if tz else None
                            start = datetime(yearExtract, monthExtract, 1, 0)  # , tzinfo=tzinfo)
                            dates = [start + timedelta(hours=h) for h in range(hours_in_month)]

                            # Extracting SLP fields, need to account for wrap around international date line
                            if self.lonLeft > self.lonRight:
                                # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                                slp1 = data.variables['Pressure_reduced_to_MSL_msl'][:,
                                       (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                                slp2 = data.variables['Pressure_reduced_to_MSL_msl'][:,
                                       (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                       0:(pos_lon2[0][0] + 1)]
                                slp = np.concatenate((slp1, slp2), axis=2)
                            else:
                                slp = data.variables['Pressure_reduced_to_MSL_msl'][:,
                                      (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                      (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]

                        else:
                            time = data.variables['valid_date_time']

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
                        if mm == 0:
                            slpCombined = slp
                            datesCombined = dates
                        else:
                            slpCombined = np.concatenate((slpCombined, slp))
                            datesCombined = np.concatenate((datesCombined, dates))


                    # are we averaging to a shorter time window?
                    m, n, p = np.shape(slpCombined)
                    slp_ = np.zeros((n * p, m))
                    grd_ = np.zeros((n * p, m))
                    for mmm in range(m):
                        slp_[:, mmm] = slpCombined[mmm, :, :].flatten()
                        vgrad = np.gradient(slpCombined[mmm, :, :])
                        grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
                    # slp_ = slp.reshape(n*m,p)

                    if self.avgTime == 0:
                        if printToScreen == True:
                            print('returning hourly values')
                        slpAvg = slp_
                        grdAvg = grd_
                        datesAvg = datesCombined
                    else:
                        numWindows = int(len(datesCombined) / self.avgTime)
                        if printToScreen == True:
                            print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                        c = 0
                        datesAvg = list()
                        slpAvg = np.empty((n * p, numWindows))
                        grdAvg = np.empty((n * p, numWindows))
                        for t in range(numWindows):
                            slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                            grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                            datesAvg.append(datesCombined[c])
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
                        if self.basin == 'atlantic':
                            inder = np.where(xFlat < 180)
                            xFlat[inder] = xFlat[inder] + 360
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
                        x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1))
                        y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1))
                    My, Mx = np.shape(y2)


                    # print('Extracted until {}-{}'.format(year2, month2))

                    from global_land_mask import globe
                    wrapLons = np.where((x2 > 180))
                    x2[wrapLons] = x2[wrapLons] - 360
                    xFlat = x2.flatten()
                    yFlat = y2.flatten()

                    isOnLandGrid = globe.is_land(y2, x2)
                    isOnLandFlat = globe.is_land(yFlat, xFlat)

                    x2[wrapLons] = x2[wrapLons] + 360
                    xFlat = x2.flatten()

                    if estelaMat == None:
                        print('you want to use ESTELA memory but did not provide a path to the data')
                    else:
                        from scipy.interpolate import RegularGridInterpolator as RGI
                        import h5py
                        with h5py.File(estelaMat, 'r') as f:
                            Xraw = f['full/Xraw'][:]
                            Y = f['full/Y'][:]
                            DJF = f['C/traveldays_interp/DJF'][:]
                            MAM = f['C/traveldays_interp/MAM'][:]
                            JJA = f['C/traveldays_interp/JJA'][:]
                            SON = f['C/traveldays_interp/SON'][:]

                        if self.basin == 'atlantic':
                            X_estela = np.copy(Xraw)
                            temp = np.where(Xraw < 40)
                            X_estela[temp] = X_estela[temp] + 360
                            X_estela = np.roll(X_estela, 440, axis=1)
                            Y_estela = np.copy(Y)
                            Y_estela = np.roll(Y_estela, 440, axis=1)
                        else:
                            X_estela = np.copy(Xraw)
                            temp = np.where(Xraw < 0)
                            X_estela[temp] = X_estela[temp] + 360
                            X_estela = np.roll(X_estela, 360, axis=1)
                            Y_estela = np.copy(Y)
                            Y_estela = np.roll(Y_estela, 360, axis=1)

                        sind = np.where(np.asarray(datesAvg) == tt)
                        datesMem = np.asarray(datesAvg)[sind[0][0]:]
                        monE = datesMem[0].month
                        if monE <= 2 or monE == 12:
                            W = DJF  # C['traveldays_interp']['DJF']
                        elif 3 <= monE <= 5:
                            W = MAM  # C['traveldays_interp']['MAM']
                        elif 6 <= monE <= 8:
                            W = JJA  # C['traveldays_interp']['JJA']
                        else:
                            W = SON  # C['traveldays_interp']['SON']

                        if self.basin == 'atlantic':
                            W2 = np.roll(W, 280, axis=1)
                        else:
                            W2 = np.roll(W, 360, axis=1)

                        points = (np.unique(Y_estela.flatten()), np.unique(X_estela.flatten()),)
                        interpF = RGI(points, W2)
                        intPoints = (y2, x2)
                        temp_interp = interpF(intPoints)
                        if self.avgTime == 0:
                            tempRounded = ceilPartial(temp_interp, 1 / 24)
                            multiplier = 24
                        else:
                            tempRounded = ceilPartial(temp_interp, self.avgTime / 24)
                            multiplier = 24 / self.avgTime

                        xvector = x2.flatten()
                        yvector = y2.flatten()
                        travelv = tempRounded.flatten()

                        slpMem = np.nan * np.ones((len(slpDownscaled), len(datesMem)))
                        grdMem = np.nan * np.ones((len(grdDownscaled), len(datesMem)))

                        for ff in range(len(datesMem)):
                            madeup_slp = np.nan * np.ones((len(travelv),))
                            madeup_grd = np.nan * np.ones((len(travelv),))
                            for hh in range(int(25 * multiplier)):
                                # print('{}'.format(hh/multiplier+1/multiplier))
                                indexIso = np.where(travelv == (hh / multiplier + 1 / multiplier))
                                madeup_slp[indexIso] = slpDownscaled[indexIso, ff + sind[0][0] - hh]
                                madeup_grd[indexIso] = grdDownscaled[indexIso, ff + sind[0][0] - hh]

                            indexIso = np.where(travelv > (hh / multiplier + 1 / multiplier))
                            madeup_slp[indexIso] = slpDownscaled[indexIso, ff + sind[0][0] - hh]
                            madeup_grd[indexIso] = grdDownscaled[indexIso, ff + sind[0][0] - hh]

                            slpMem[:, ff] = madeup_slp
                            grdMem[:, ff] = madeup_grd

                        trimSlps = slpMem[~isOnLandFlat, :]
                        trimGrds = grdMem[~isOnLandFlat, :]

                        if counter == 0:
                            SLPS = trimSlps
                            GRDS = trimGrds
                            DATES = datesMem
                        else:
                            SLPS = np.hstack((SLPS, trimSlps))
                            GRDS = np.hstack((GRDS, trimGrds))
                            DATES = np.append(DATES, datesMem)

                        counter = counter + 1

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

            with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)


    def extractCFSRLocal(self,centralNode,printToScreen=False,estelaMat=None,loadPrior=False,loadPickle='./'):
        '''
        This function is utilized for opening *.raw Argus files prior to a debayering task,
        and adds the

        Must call this function before calling any debayering functions
        '''

        if loadPrior==True:
            import pickle
            with open(loadPickle, "rb") as input_file:
                slps = pickle.load(input_file)
            self.DATESLocal = slps['DATESLocal']
            self.xGridLocal = slps['x2Local']
            self.yGridLocal = slps['y2Local']
            self.xFlatLocal = slps['xFlatLocal']
            self.yFlatLocal = slps['yFlatLocal']
            self.isOnLandGridLocal = slps['isOnLandGridLocal']
            self.isOnLandFlatLocal = slps['isOnLandFlatLocal']
            self.SLPSLocal = slps['SLPSLocal']
            self.GRDSLocal = slps['GRDSLocal']
            self.MxLocal = slps['MxLocal']
            self.MyLocal = slps['MyLocal']
            print('loaded prior SLP Local predictor')

        else:


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

            def ceilPartial(value, resolution):
                return np.ceil(value / resolution) * resolution

            def floorPartial(value, resolution):
                return np.floor(value / resolution) * resolution

            # if centralNode[1] < 0:
            #     centralNode[1]=centralNode[1]+360

            print(centralNode[0], centralNode[1])
            print(floorPartial(centralNode[1],1), ceilPartial(centralNode[1],1), ceilPartial(centralNode[0],1), floorPartial(centralNode[0],1))

            lon_left = floorPartial(centralNode[1],1) - 2
            lon_right = ceilPartial(centralNode[1],1) + 2

            pos_lon1 = np.where((lon == floorPartial(centralNode[1],1) - 2))
            pos_lon2 = np.where((lon == ceilPartial(centralNode[1],1) + 2))

            pos_lat2 = np.where((lat == ceilPartial(centralNode[0],1) + 2))
            pos_lat1 = np.where((lat == floorPartial(centralNode[0],1) - 2))
            print(pos_lon1, pos_lon2, pos_lat1, pos_lat2)

            latitud = lat[(pos_lat2[0][0]):(pos_lat1[0][0] + 1)]
            if lon_left > lon_right:
                longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
            else:
                longitud = lon[pos_lon1[0][0]:(pos_lon2[0][0] + 1)]
            [x, y] = np.meshgrid(longitud, latitud)

            def roundPartial(value, resolution):
                return round(value / resolution) * resolution


            counter = 0
            # Now need to loop through the number of files we're extracting data from
            for tt in extractTime:

                if self.localMemory == False:
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

                    if printToScreen == True:
                        print('{}-{}'.format(yearExtract, monthExtract))


                    if (yearExtract == 2025 and monthExtract >= 1) or yearExtract > 2025:
                        time = data.variables['time']
                        from datetime import timedelta
                        from calendar import monthrange
                        import zoneinfo  # Python ≥3.9; otherwise use `pytz`
                        #def hourly_vector_stdlib(year: int, month: int, tz: str | None = None):
                        hours_in_month = monthrange(yearExtract, monthExtract)[1] * 24
                        #tzinfo = zoneinfo.ZoneInfo(tz) if tz else None
                        start = datetime(yearExtract, monthExtract, 1, 0)#, tzinfo=tzinfo)
                        dates = [start + timedelta(hours=h) for h in range(hours_in_month)]

                        # Extracting SLP fields, need to account for wrap around international date line
                        if lon_left > lon_right:
                            # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                            slp1 = data.variables['Pressure_reduced_to_MSL_msl'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                            slp2 = data.variables['Pressure_reduced_to_MSL_msl'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                   0:(pos_lon2[0][0] + 1)]
                            slp = np.concatenate((slp1, slp2), axis=2)
                        else:
                            slp = data.variables['Pressure_reduced_to_MSL_msl'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                  (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]
                    else:
                        time = data.variables['valid_date_time']
                        # extract times and turn them into datetimes
                        # years = np.empty((len(time),))
                        year = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[0:4]) for i in time]
                        month = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[4:6]) for i in time]
                        day = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[6:8]) for i in time]
                        hour = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[8:]) for i in time]
                        d_vec = np.vstack((year, month, day, hour)).T
                        dates = [datetime(d[0], d[1], d[2], d[3], 0, 0) for d in d_vec]

                        # Extracting SLP fields, need to account for wrap around international date line
                        if lon_left > lon_right:
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
                    if self.resolutionLocal == 0.5:
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


                        if self.basin == 'atlantic':
                            inder = np.where(xFlat < 180)
                            xFlat[inder] = xFlat[inder] + 360


                        xRem = np.fmod(xFlat, self.resolutionLocal)
                        yRem = np.fmod(yFlat, self.resolutionLocal)
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

                    # print('Extracted until {}-{}'.format(year2, month2))

                    # FOR THE LOCAL WE WANT TO KEEP LAND VALUES?
                    from global_land_mask import globe
                    wrapLons = np.where((x2 > 180))
                    x2[wrapLons] = x2[wrapLons] - 360
                    xFlat = x2.flatten()
                    yFlat = y2.flatten()

                    isOnLandGrid = globe.is_land(y2, x2)
                    isOnLandFlat = globe.is_land(yFlat, xFlat)

                    x2[wrapLons] = x2[wrapLons] + 360
                    xFlat = x2.flatten()

                    trimSlps = slpMem#[~isOnLandFlat, :]
                    trimGrds = grdMem#[~isOnLandFlat, :]

                    if counter == 0:
                        SLPS = trimSlps
                        GRDS = trimGrds
                        DATES = datesMem
                    else:
                        SLPS = np.hstack((SLPS, trimSlps))
                        GRDS = np.hstack((GRDS, trimGrds))
                        DATES = np.append(DATES, datesMem)

                    counter = counter + 1

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
                        print('{}-{}'.format(yearExtract, monthExtract))

                        if (yearExtract == 2025 and monthExtract >= 1) or yearExtract > 2025:
                            time = data.variables['time']
                            from datetime import timedelta
                            from calendar import monthrange
                            import zoneinfo  # Python ≥3.9; otherwise use `pytz`
                            # def hourly_vector_stdlib(year: int, month: int, tz: str | None = None):
                            hours_in_month = monthrange(yearExtract, monthExtract)[1] * 24
                            # tzinfo = zoneinfo.ZoneInfo(tz) if tz else None
                            start = datetime(yearExtract, monthExtract, 1, 0)  # , tzinfo=tzinfo)
                            dates = [start + timedelta(hours=h) for h in range(hours_in_month)]

                            # Extracting SLP fields, need to account for wrap around international date line
                            if self.lonLeft > self.lonRight:
                                # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                                slp1 = data.variables['Pressure_reduced_to_MSL_msl'][:,
                                       (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                                slp2 = data.variables['Pressure_reduced_to_MSL_msl'][:,
                                       (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                       0:(pos_lon2[0][0] + 1)]
                                slp = np.concatenate((slp1, slp2), axis=2)
                            else:
                                slp = data.variables['Pressure_reduced_to_MSL_msl'][:,
                                      (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                                      (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]

                        else:
                            time = data.variables['valid_date_time']

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
                        if mm == 0:
                            slpCombined = slp
                            datesCombined = dates
                        else:
                            slpCombined = np.concatenate((slpCombined, slp))
                            datesCombined = np.concatenate((datesCombined, dates))


                    # are we averaging to a shorter time window?
                    m, n, p = np.shape(slpCombined)
                    slp_ = np.zeros((n * p, m))
                    grd_ = np.zeros((n * p, m))
                    for mmm in range(m):
                        slp_[:, mmm] = slpCombined[mmm, :, :].flatten()
                        vgrad = np.gradient(slpCombined[mmm, :, :])
                        grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
                    # slp_ = slp.reshape(n*m,p)

                    if self.avgTime == 0:
                        if printToScreen == True:
                            print('returning hourly values')
                        slpAvg = slp_
                        grdAvg = grd_
                        datesAvg = datesCombined
                    else:
                        numWindows = int(len(datesCombined) / self.avgTime)
                        if printToScreen == True:
                            print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                        c = 0
                        datesAvg = list()
                        slpAvg = np.empty((n * p, numWindows))
                        grdAvg = np.empty((n * p, numWindows))
                        for t in range(numWindows):
                            slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                            grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                            datesAvg.append(datesCombined[c])
                            c = c + self.avgTime

                    # are we reducing the resolution of the grid?
                    if self.resolutionLocal == 0.5:
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
                        if self.basin == 'atlantic':
                            inder = np.where(xFlat < 180)
                            xFlat[inder] = xFlat[inder] + 360
                        xRem = np.fmod(xFlat, self.resolutionLocal)
                        yRem = np.fmod(yFlat, self.resolutionLocal)
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
                        x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1))
                        y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1))
                    My, Mx = np.shape(y2)


                    # print('Extracted until {}-{}'.format(year2, month2))

                    from global_land_mask import globe
                    wrapLons = np.where((x2 > 180))
                    x2[wrapLons] = x2[wrapLons] - 360
                    xFlat = x2.flatten()
                    yFlat = y2.flatten()

                    isOnLandGrid = globe.is_land(y2, x2)
                    isOnLandFlat = globe.is_land(yFlat, xFlat)

                    x2[wrapLons] = x2[wrapLons] + 360
                    xFlat = x2.flatten()

                    if estelaMat == None:
                        print('you want to use ESTELA memory but did not provide a path to the data')
                    else:
                        from scipy.interpolate import RegularGridInterpolator as RGI
                        import h5py
                        with h5py.File(estelaMat, 'r') as f:
                            Xraw = f['full/Xraw'][:]
                            Y = f['full/Y'][:]
                            DJF = f['C/traveldays_interp/DJF'][:]
                            MAM = f['C/traveldays_interp/MAM'][:]
                            JJA = f['C/traveldays_interp/JJA'][:]
                            SON = f['C/traveldays_interp/SON'][:]

                        if self.basin == 'atlantic':
                            X_estela = np.copy(Xraw)
                            temp = np.where(Xraw < 40)
                            X_estela[temp] = X_estela[temp] + 360
                            X_estela = np.roll(X_estela, 440, axis=1)
                            Y_estela = np.copy(Y)
                            Y_estela = np.roll(Y_estela, 440, axis=1)
                        else:
                            X_estela = np.copy(Xraw)
                            temp = np.where(Xraw < 0)
                            X_estela[temp] = X_estela[temp] + 360
                            X_estela = np.roll(X_estela, 360, axis=1)
                            Y_estela = np.copy(Y)
                            Y_estela = np.roll(Y_estela, 360, axis=1)

                        sind = np.where(np.asarray(datesAvg) == tt)
                        datesMem = np.asarray(datesAvg)[sind[0][0]:]
                        monE = datesMem[0].month
                        if monE <= 2 or monE == 12:
                            W = DJF  # C['traveldays_interp']['DJF']
                        elif 3 <= monE <= 5:
                            W = MAM  # C['traveldays_interp']['MAM']
                        elif 6 <= monE <= 8:
                            W = JJA  # C['traveldays_interp']['JJA']
                        else:
                            W = SON  # C['traveldays_interp']['SON']

                        if self.basin == 'atlantic':
                            W2 = np.roll(W, 280, axis=1)
                        else:
                            W2 = np.roll(W, 360, axis=1)

                        points = (np.unique(Y_estela.flatten()), np.unique(X_estela.flatten()),)
                        interpF = RGI(points, W2)
                        intPoints = (y2, x2)
                        temp_interp = interpF(intPoints)
                        if self.avgTime == 0:
                            tempRounded = ceilPartial(temp_interp, 1 / 24)
                            multiplier = 24
                        else:
                            tempRounded = ceilPartial(temp_interp, self.avgTime / 24)
                            multiplier = 24 / self.avgTime

                        xvector = x2.flatten()
                        yvector = y2.flatten()
                        travelv = tempRounded.flatten()

                        slpMem = np.nan * np.ones((len(slpDownscaled), len(datesMem)))
                        grdMem = np.nan * np.ones((len(grdDownscaled), len(datesMem)))

                        for ff in range(len(datesMem)):
                            madeup_slp = np.nan * np.ones((len(travelv),))
                            madeup_grd = np.nan * np.ones((len(travelv),))
                            for hh in range(int(25 * multiplier)):
                                # print('{}'.format(hh/multiplier+1/multiplier))
                                indexIso = np.where(travelv == (hh / multiplier + 1 / multiplier))
                                madeup_slp[indexIso] = slpDownscaled[indexIso, ff + sind[0][0] - hh]
                                madeup_grd[indexIso] = grdDownscaled[indexIso, ff + sind[0][0] - hh]

                            indexIso = np.where(travelv > (hh / multiplier + 1 / multiplier))
                            madeup_slp[indexIso] = slpDownscaled[indexIso, ff + sind[0][0] - hh]
                            madeup_grd[indexIso] = grdDownscaled[indexIso, ff + sind[0][0] - hh]

                            slpMem[:, ff] = madeup_slp
                            grdMem[:, ff] = madeup_grd

                        trimSlps = slpMem[~isOnLandFlat, :]
                        trimGrds = grdMem[~isOnLandFlat, :]

                        if counter == 0:
                            SLPS = trimSlps
                            GRDS = trimGrds
                            DATES = datesMem
                        else:
                            SLPS = np.hstack((SLPS, trimSlps))
                            GRDS = np.hstack((GRDS, trimGrds))
                            DATES = np.append(DATES, datesMem)

                        counter = counter + 1

            self.xGridLocal = x2
            self.yGridLocal = y2
            self.xFlatLocal = xFlat
            self.yFlatLocal = yFlat
            self.isOnLandGridLocal = isOnLandGrid
            self.isOnLandFlatLocal = isOnLandFlat
            self.SLPSLocal = SLPS
            self.GRDSLocal = GRDS
            self.DATESLocal = DATES
            self.MxLocal = Mx
            self.MyLocal = My

            import pickle
            samplesPickle = 'slpsLocal.pickle'
            outputSamples = {}
            outputSamples['x2Local'] = x2
            outputSamples['y2Local'] = y2
            outputSamples['xFlatLocal'] = xFlat
            outputSamples['yFlatLocal'] = yFlat
            outputSamples['isOnLandGridLocal'] = isOnLandGrid
            outputSamples['isOnLandFlatLocal'] = isOnLandFlat
            outputSamples['SLPSLocal'] = SLPS
            outputSamples['GRDSLocal'] = GRDS
            outputSamples['DATESLocal'] = DATES
            outputSamples['MxLocal'] = Mx
            outputSamples['MyLocal'] = My

            with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)


    def pcaOfSlps(self,loadPrior=False,loadPickle='./'):

        if loadPrior==True:
            import pickle
            with open(loadPickle, "rb") as input_file:
                pcas = pickle.load(input_file)
            self.SlpGrdMean = pcas['SlpGrdMean']
            self.SlpGrdStd = pcas['SlpGrdStd']
            self.SlpGrdNorm = pcas['SlpGrdNorm']
            self.SlpGrd = pcas['SlpGrd']
            self.PCs = pcas['PCs']
            self.EOFs = pcas['EOFs']
            self.variance = pcas['variance']
            self.nPercent = pcas['nPercent']
            self.APEV = pcas['APEV']
            self.nterm = pcas['nterm']
            print('loaded prior PCA processing')


        else:

            from sklearn.decomposition import PCA

            trimSlps = self.SLPS
            trimGrds = self.GRDS

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

            with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)



    def pcaOfSlpsLocal(self,loadPrior=False,loadPickle='./'):

        if loadPrior==True:
            import pickle
            with open(loadPickle, "rb") as input_file:
                pcas = pickle.load(input_file)
            self.SlpGrdMeanLocal = pcas['SlpGrdMeanLocal']
            self.SlpGrdStdLocal = pcas['SlpGrdStdLocal']
            self.SlpGrdNormLocal = pcas['SlpGrdNormLocal']
            self.SlpGrdLocal = pcas['SlpGrdLocal']
            self.PCsLocal = pcas['PCsLocal']
            self.EOFsLocal = pcas['EOFsLocal']
            self.varianceLocal = pcas['varianceLocal']
            self.nPercentLocal = pcas['nPercentLocal']
            self.APEVLocal = pcas['APEVLocal']
            self.ntermLocal = pcas['ntermLocal']
            print('loaded prior PCA processing')


        else:

            from sklearn.decomposition import PCA

            trimSlps = self.SLPSLocal
            trimGrds = self.GRDSLocal

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

            self.SlpGrdMeanLocal = SlpGrdMean
            self.SlpGrdStdLocal = SlpGrdStd
            self.SlpGrdNormLocal = SlpGrdNorm
            self.SlpGrdLocal = SlpGrd
            self.PCsLocal = PCs
            self.EOFsLocal = EOFs
            self.varianceLocal = variance
            self.nPercentLocal = nPercent
            self.APEVLocal = APEV
            self.ntermLocal = nterm

            import pickle
            samplesPickle = 'pcasLocal.pickle'
            outputSamples = {}
            outputSamples['SlpGrdMeanLocal'] = SlpGrdMean
            outputSamples['SlpGrdStdLocal'] = SlpGrdStd
            outputSamples['SlpGrdNormLocal'] = SlpGrdNorm
            outputSamples['SlpGrdLocal'] = SlpGrd
            outputSamples['PCsLocal'] = PCs
            outputSamples['EOFsLocal'] = EOFs
            outputSamples['varianceLocal'] = variance
            outputSamples['nPercentLocal'] = nPercent
            outputSamples['APEVLocal'] = APEV
            outputSamples['ntermLocal'] = nterm

            with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)



    def wtClusters(self,numClusters=49,TCs=True,Basin=b'NA',RG=None,minGroupSize=50,alphaRG=0.1,met=None,loadPrior=False,loadPickle='./'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                dwts = pickle.load(input_file)

            self.Km_ETC = dwts['Km_ETC']
            self.bmus = dwts['bmus']
            self.sorted_centroidsETC = dwts['sorted_centroidsETC']
            self.sorted_cenEOFsETC = dwts['sorted_cenEOFsETC']
            self.bmus_correctedETC = dwts['bmus_correctedETC']
            self.kmaOrderETC = dwts['kmaOrderETC']
            self.dGroupsETC = dwts['dGroupsETC']
            self.groupSizeETC = dwts['groupSizeETC']
            self.numClustersETC = dwts['numClustersETC']
            self.bmus_corrected = dwts['bmus_corrected']
            self.windowHs = dwts['windowHs']
            self.windowTp = dwts['windowTp']
            self.numClusters = np.nanmax(dwts['numClustersETC'])

            print('loaded prior DWT processing')

        else:

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
                self.tcData = data
                tcBasin = data['basin']
                TCtime = data['time'].values
                # TClon = data['lon'].values
                # TClat = data['lat'].values
                # TCpres = data['usa_pres']
                # TCwind = data['usa_wind']
                import numpy as np

                print('isolating your ocean basin: {}'.format(self.basin))

                indexTC = np.where(tcBasin[:,0]==self.basin)
                tcTime = TCtime[indexTC[0],:]
                # tcPres = TCpres[indexTC[0],:]
                # tcWind = TCwind[indexTC[0],:]
                # tcLon = TClon[indexTC[0],:]
                # tcLat = TClat[indexTC[0],:]

                from functions import dt2cal
                tcAllTime = [dt2cal(dt) for dt in tcTime.flatten()]
                if self.avgTime == 24:
                    tcDailyTime = [np.array([d[0],d[1],d[2]]) for d in tcAllTime]
                else:
                    tcDailyTime = np.vstack(
                        [np.array([d[0], d[1], d[2], (np.floor_divide(d[3], 12) * 12), 0, 0]) for d in tcAllTime])

                # recentTCs = np.where(tcHalfDayTime[:, 0] > 1940)
                # allTCtimes = tcHourlyTime[recentTCs[0], :]
                # allTCLat = tcLat[recentTCs[0]]
                # allTCLon = tcLon[recentTCs[0]]
                # allTCPres = tcPres[recentTCs[0]]
                # allTCHalfDayTime = tcHalfDayTime[recentTCs[0], :]
                #
                # # u, idx, counts = np.unique(tcAllTime, axis=0, return_index=True, return_counts=True)
                # allTCtimes = np.unique(allTCHalfDayTime, axis=0, return_index=False, return_counts=False)

                # u, idx, counts = np.unique(tcAllTime, axis=0, return_index=True, return_counts=True)
                allTCtimes = np.unique(tcDailyTime, axis=0, return_index=False, return_counts=False)
                recentTCs = np.where(allTCtimes[:,0] > 1940)
                allTCtimesVec = allTCtimes[recentTCs[0],:]

                import datetime
                def dateDay2datetime(d_vec):
                    '''
                    Returns datetime list from a datevec matrix
                    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
                    '''
                    return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]

                def dateDayHour2datetime(d_vec):
                    '''
                    Returns datetime list from a datevec matrix
                    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
                    '''
                    return [datetime.datetime(d[0], d[1], d[2], d[3], 0) for d in d_vec]

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

                allTCtimes = np.asarray(dateDayHour2datetime(allTCtimesVec))
                # import pandas as pd
                # df = pd.DataFrame(allTCtimes, columns=['date'])
                # dropDups = df.drop_duplicates('date')

                tcDates = allTCtimes#dropDups['date'].dt.date.tolist()
                slpDates = self.DATES#datetime2datetimeDate(self.DATES)
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
                self.tcDates = tcDates
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
                outputSamples['tcData'] = self.tcData
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

                with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                    pickle.dump(outputSamples, f)




            else:

                if RG == 'waves':
                    self.minGroupSize = minGroupSize
                    windowHs = []
                    windowTp = []
                    for qq in range(len(self.DATES)-1):
                        if np.remainder(qq,5000) == 0:
                            print('done up to {}'.format(self.DATES[qq]))
                        windowIndex = np.where((met.timeWave > self.DATES[qq]) & (met.timeWave < self.DATES[qq+1]))
                        if len(windowIndex[0])>0:
                            windowHs.append(np.max(met.Hs[windowIndex]))
                            windowTp.append(np.mean(met.Tp[windowIndex]))
                        else:
                            windowHs.append(np.nan)
                            windowTp.append(np.nan)

                    self.windowHs = np.asarray(windowHs)
                    self.windowTp = np.asarray(windowTp)

                    nanInds = np.where(~np.isnan(windowHs))
                    # PCsub = PCsub[nanInds[0],:]
                    PCsub_std = np.std(PCsub[nanInds[0],:], axis=0)
                    PCsub_norm = np.divide(PCsub[nanInds[0],:], PCsub_std)
                    PCsub_std_pred = np.std(PCsub, axis=0)
                    PCsub_norm_pred = np.divide(PCsub, PCsub_std_pred)
                    X_fit = PCsub_norm  #  predictor
                    X_pred = PCsub_norm_pred  #  predictor

                    wd = np.vstack((np.asarray(windowHs)[nanInds], np.asarray(windowTp)[nanInds])).T
                    wd_std = np.nanstd(wd, axis=0)
                    wd_norm = np.divide(wd, wd_std)

                    Y = wd_norm  # predictand

                    # Adjust
                    [n, d] = Y.shape
                    X_fit = np.concatenate((np.ones((n, 1)), X_fit), axis=1)
                    [n2, d2] = X_pred.shape
                    X_pred = np.concatenate((np.ones((n2, 1)), X_pred), axis=1)

                    from sklearn import linear_model

                    clf = linear_model.LinearRegression(fit_intercept=True)
                    Ymod = np.zeros((n2, d)) * np.nan
                    for i in range(d):
                        clf.fit(X_fit, Y[:, i])
                        beta = clf.coef_
                        intercept = clf.intercept_
                        Ymod[:, i] = np.ones((n2,)) * intercept
                        for j in range(len(beta)):
                            Ymod[:, i] = Ymod[:, i] + beta[j] * X_pred[:, j]

                    # de-scale
                    Ym = np.multiply(Ymod, wd_std)

                    alpha = alphaRG
                    # append Yregres data to PCs
                    data = np.concatenate((PCsub, Ym), axis=1)
                    data_std = np.std(data, axis=0)
                    data_mean = np.mean(data, axis=0)

                    #  normalize but keep PCs weigth
                    data_norm = np.ones(data.shape) * np.nan
                    for i in range(PCsub.shape[1]):
                        data_norm[:, i] = np.divide(data[:, i] - data_mean[i], data_std[0])
                    for i in range(PCsub.shape[1], data.shape[1]):
                        data_norm[:, i] = np.divide(data[:, i] - data_mean[i], data_std[i])

                    # apply alpha (PCs - Yregress weight)
                    data_a = np.concatenate(
                        ((1 - alpha) * data_norm[:, :self.nterm],
                         alpha * data_norm[:, self.nterm:]),
                        axis=1
                    )

                    #  KMeans
                    keep_iter = True
                    count_iter = 0
                    while keep_iter:
                        # n_init: number of times KMeans runs with different centroids seeds
                        kma = KMeans(n_clusters=self.numClusters, n_init=100).fit(data_a)

                        #  check minimun group_size
                        group_keys, group_size = np.unique(kma.labels_, return_counts=True)

                        # sort output
                        group_k_s = np.column_stack([group_keys, group_size])
                        group_k_s = group_k_s[group_k_s[:, 0].argsort()]  # sort by cluster num

                        if not self.minGroupSize:
                            keep_iter = False

                        else:
                            # keep iterating?
                            keep_iter = np.where(group_k_s[:, 1] < self.minGroupSize)[0].any()
                            count_iter += 1

                            # log kma iteration
                            print('KMA iteration info:')
                            for rr in group_k_s:
                                print('  cluster: {0}, size: {1}'.format(rr[0], rr[1]))
                            print('Try again: ', keep_iter)
                            print('Total attemps: ', count_iter)
                            print()



                    # groupsize
                    _, group_sizeETC = np.unique(kma.labels_, return_counts=True)
                    # groups
                    d_groupsETC = {}
                    for k in range(numClusters):
                        d_groupsETC['{0}'.format(k)] = np.where(kma.labels_ == k)
                    self.groupSizeETC = group_sizeETC
                    self.dGroupsETC = d_groupsETC
                    # centroids
                    # centroids = np.dot(kma.cluster_centers_, EOFsub)

                    centroids = np.zeros((self.numClusters, EOFsub.shape[1]))  # PCsub.shape[1]))
                    for k in range(self.numClusters):
                        centroids[k, :] = np.dot(np.mean(PCsub[d_groupsETC['{0}'.format(k)], :], axis=1), EOFsub)


                    # km, x and var_centers
                    km = np.multiply(
                        centroids,
                        np.tile(self.SlpGrdStd, (self.numClusters, 1))
                    ) + np.tile(self.SlpGrdMean, (self.numClusters, 1))
                    # sort kmeans
                    kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, self.numClusters)
                    self.kmaOrderETC = kma_order
                    bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
                    for i in range(self.numClusters):
                        posc = np.where(kma.labels_ == kma_order[i])
                        bmus_corrected[posc] = i
                    self.bmus_correctedETC = bmus_corrected
                    # reorder centroids
                    self.sorted_cenEOFsETC = kma.cluster_centers_[kma_order, :]
                    self.sorted_centroidsETC = centroids[kma_order, :]

                    repmatStd = np.tile(self.SlpGrdStd, (numClusters, 1))
                    repmatMean = np.tile(self.SlpGrdMean, (numClusters, 1))
                    self.Km_ETC = np.multiply(self.sorted_centroidsETC, repmatStd) + repmatMean
                    self.bmus = kma.labels_#self.bmus_correctedETC
                    self.bmus_corrected = self.bmus_correctedETC


                elif RG == 'seasonal':
                    self.minGroupSize=minGroupSize
                    windowHs = []
                    windowTp = []
                    for qq in range(len(self.DATES)-1):
                        if np.remainder(qq,5000) == 0:
                            print('done up to {}'.format(self.DATES[qq]))
                        windowIndex = np.where((met.timeWave > self.DATES[qq]) & (met.timeWave < self.DATES[qq+1]))
                        windowHs.append(np.max(met.Hs[windowIndex]))
                        windowTp.append(np.mean(met.Tp[windowIndex]))

                    self.windowHs = np.asarray(windowHs)
                    self.windowTp = np.asarray(windowTp)

                    dayOfYear = np.array([hh.timetuple().tm_yday for hh in self.DATES])  # returns 1 for January 1st
                    dayOfYearSine = np.sin(2 * np.pi / 366 * dayOfYear)
                    dayOfYearCosine = np.cos(2 * np.pi / 366 * dayOfYear)

                    PCsub_std = np.std(PCsub, axis=0)
                    PCsub_norm = np.divide(PCsub, PCsub_std)

                    X = PCsub_norm  #  predictor

                    wd = np.vstack((dayOfYearSine, dayOfYearCosine)).T

                    wd_std = np.nanstd(wd, axis=0)
                    wd_norm = np.divide(wd, wd_std)

                    Y = wd_norm  # predictand

                    # Adjust
                    [n, d] = Y.shape
                    X = np.concatenate((np.ones((n, 1)), X), axis=1)
                    from sklearn import linear_model

                    clf = linear_model.LinearRegression(fit_intercept=True)
                    Ymod = np.zeros((n, d)) * np.nan
                    for i in range(d):
                        clf.fit(X, Y[:, i])
                        beta = clf.coef_
                        intercept = clf.intercept_
                        Ymod[:, i] = np.ones((n,)) * intercept
                        for j in range(len(beta)):
                            Ymod[:, i] = Ymod[:, i] + beta[j] * X[:, j]

                    # de-scale
                    Ym = np.multiply(Ymod, wd_std)

                    alpha = alphaRG
                    # append Yregres data to PCs
                    data = np.concatenate((PCsub, Ym), axis=1)
                    data_std = np.std(data, axis=0)
                    data_mean = np.mean(data, axis=0)

                    #  normalize but keep PCs weigth
                    data_norm = np.ones(data.shape) * np.nan
                    for i in range(PCsub.shape[1]):
                        data_norm[:, i] = np.divide(data[:, i] - data_mean[i], data_std[0])
                    for i in range(PCsub.shape[1], data.shape[1]):
                        data_norm[:, i] = np.divide(data[:, i] - data_mean[i], data_std[i])

                    # apply alpha (PCs - Yregress weight)
                    data_a = np.concatenate(
                        ((1 - alpha) * data_norm[:, :self.nterm],
                         alpha * data_norm[:, self.nterm:]),
                        axis=1
                    )

                    #  KMeans
                    keep_iter = True
                    count_iter = 0
                    while keep_iter:
                        # n_init: number of times KMeans runs with different centroids seeds
                        kma = KMeans(n_clusters=self.numClusters, n_init=100).fit(data_a)

                        #  check minimun group_size
                        group_keys, group_size = np.unique(kma.labels_, return_counts=True)

                        # sort output
                        group_k_s = np.column_stack([group_keys, group_size])
                        group_k_s = group_k_s[group_k_s[:, 0].argsort()]  # sort by cluster num

                        if not self.minGroupSize:
                            keep_iter = False

                        else:
                            # keep iterating?
                            keep_iter = np.where(group_k_s[:, 1] < self.minGroupSize)[0].any()
                            count_iter += 1

                            # log kma iteration
                            print('KMA iteration info:')
                            for rr in group_k_s:
                                print('  cluster: {0}, size: {1}'.format(rr[0], rr[1]))
                            print('Try again: ', keep_iter)
                            print('Total attemps: ', count_iter)
                            print()



                    # groupsize
                    _, group_sizeETC = np.unique(kma.labels_, return_counts=True)
                    # groups
                    d_groupsETC = {}
                    for k in range(numClusters):
                        d_groupsETC['{0}'.format(k)] = np.where(kma.labels_ == k)
                    self.groupSizeETC = group_sizeETC
                    self.dGroupsETC = d_groupsETC
                    # centroids
                    # centroids = np.dot(kma.cluster_centers_, EOFsub)

                    centroids = np.zeros((self.numClusters, EOFsub.shape[1]))  # PCsub.shape[1]))
                    for k in range(self.numClusters):
                        centroids[k, :] = np.dot(np.mean(PCsub[d_groupsETC['{0}'.format(k)], :], axis=1), EOFsub)


                    # km, x and var_centers
                    km = np.multiply(
                        centroids,
                        np.tile(self.SlpGrdStd, (self.numClusters, 1))
                    ) + np.tile(self.SlpGrdMean, (self.numClusters, 1))
                    # sort kmeans
                    kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, self.numClusters)
                    self.kmaOrderETC = kma_order
                    bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
                    for i in range(self.numClusters):
                        posc = np.where(kma.labels_ == kma_order[i])
                        bmus_corrected[posc] = i
                    self.bmus_correctedETC = bmus_corrected
                    # reorder centroids
                    self.sorted_cenEOFsETC = kma.cluster_centers_[kma_order, :]
                    self.sorted_centroidsETC = centroids[kma_order, :]

                    repmatStd = np.tile(self.SlpGrdStd, (numClusters, 1))
                    repmatMean = np.tile(self.SlpGrdMean, (numClusters, 1))
                    self.Km_ETC = np.multiply(self.sorted_centroidsETC, repmatStd) + repmatMean
                    self.bmus = self.bmus_correctedETC
                    self.bmus_corrected = self.bmus_correctedETC


                elif RG == 'default':
                    print('performaing a regular K-Means clustering')
                    # kma = KMeans(n_clusters=numClusters, n_init=2000).fit(PCsub)

                    #  KMeans
                    keep_iter = True
                    count_iter = 0
                    while keep_iter:
                        # n_init: number of times KMeans runs with different centroids seeds
                        kma = KMeans(n_clusters=self.numClusters, n_init=100).fit(PCsub)

                        #  check minimun group_size
                        group_keys, group_size = np.unique(kma.labels_, return_counts=True)

                        # sort output
                        group_k_s = np.column_stack([group_keys, group_size])
                        group_k_s = group_k_s[group_k_s[:, 0].argsort()]  # sort by cluster num

                        if not self.minGroupSize:
                            keep_iter = False

                        else:
                            # keep iterating?
                            keep_iter = np.where(group_k_s[:, 1] < self.minGroupSize)[0].any()
                            count_iter += 1

                            # log kma iteration
                            print('KMA iteration info:')
                            for rr in group_k_s:
                                print('  cluster: {0}, size: {1}'.format(rr[0], rr[1]))
                            print('Try again: ', keep_iter)
                            print('Total attemps: ', count_iter)
                            print()



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
                    self.bmus = self.bmus_correctedETC
                    self.bmus_corrected = self.bmus_correctedETC
                    self.windowHs = np.nan
                    self.windowTp = np.nan

                import pickle
                samplesPickle = 'dwts.pickle'
                outputSamples = {}
                outputSamples['Km_ETC'] = self.Km_ETC
                outputSamples['bmus'] = self.bmus
                outputSamples['sorted_centroidsETC'] = self.sorted_centroidsETC
                outputSamples['sorted_cenEOFsETC'] = self.sorted_cenEOFsETC
                outputSamples['bmus_correctedETC'] = self.bmus_correctedETC
                outputSamples['kmaOrderETC'] = self.kmaOrderETC
                outputSamples['dGroupsETC'] = self.dGroupsETC
                outputSamples['groupSizeETC'] = self.groupSizeETC
                outputSamples['numClustersETC'] = self.numClusters
                outputSamples['numClusters'] = self.numClusters
                outputSamples['bmus_corrected'] = self.bmus_correctedETC
                outputSamples['windowHs'] = self.windowHs
                outputSamples['windowTp'] = self.windowTp

                with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                    pickle.dump(outputSamples, f)



    def alrSimulations(self,climate,historicalSimNum,futureSimNum,futureSimStart,futureSimEnd,plotOutput=False,loadPrior=False,loadPickle='./'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                simdwts = pickle.load(input_file)

            self.futureBmusSim = simdwts['futureBmusSim']
            self.futureDatesSim = simdwts['futureDatesSim']
            self.historicalDatesSim = simdwts['historicalDatesSim']
            self.historicalBmusSim = simdwts['historicalBmusSim']
            self.simHistBmuLengthChopped = simdwts['simHistBmuLengthChopped']
            self.simHistBmuGroupsChopped = simdwts['simHistBmuGroupsChopped']
            self.simHistBmuChopped = simdwts['simHistBmuChopped']
            self.simFutureBmuLengthChopped = simdwts['simFutureBmuLengthChopped']
            self.simFutureBmuGroupsChopped = simdwts['simFutureBmuGroupsChopped']
            self.simFutureBmuChopped = simdwts['simFutureBmuChopped']
            self.futureSimEnd = simdwts['futureSimEnd']
            self.futureSimStart = simdwts['futureSimStart']

            print('loaded prior ALR DWT processing')

        else:
            from functions import xds_reindex_daily as xr_daily
            from functions import xds_common_dates_daily as xcd_daily
            from functions import ALR_WRP
            from functions import xds_reindex_flexible as xr_flexible
            from functions import xds_common_dates_flexible as xcd_flexible
            import numpy as np
            from datetime import datetime, timedelta
            self.xds_KMA_fit = xr.Dataset(
                {
                    'bmus': (('time',), self.bmus_corrected+1),
                },
                coords={'time': [x for x in self.DATES]}
            )
            # self.xds_KMA_fit = xr_daily(self.xds_KMA_fit, datetime(1979, 6, 1), datetime(2024, 5, 31))
            self.xds_KMA_fit = xr_flexible(self.xds_KMA_fit, datetime(1979, 6, 1), datetime(2024, 5, 31),avgTime=self.avgTime)



            # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
            self.xds_PCs_fit = xr.Dataset(
                {
                    'PC1': (('time',), climate.dailyPC1[0:len(climate.mjoYear)]),
                    'PC2': (('time',), climate.dailyPC3[0:len(climate.mjoYear)]),
                    'PC3': (('time',), climate.dailyPC3[0:len(climate.mjoYear)]),
                },
                coords={'time': [datetime(climate.mjoYear[r], climate.mjoMonth[r], climate.mjoDay[r]) for r in range(len(climate.mjoDay))]}
            )
            # reindex annual data to daily data
            # self.xds_PCs_fit = xr_daily(self.xds_PCs_fit, datetime(1979, 6, 1), datetime(2024, 5, 31))
            self.xds_PCs_fit = xr_flexible(self.xds_PCs_fit, datetime(1979, 6, 1), datetime(2024, 5, 31),avgTime=self.avgTime)

            # MJO: RMM1s (Generated with copula simulation. Annual data, parse to daily)
            self.xds_MJO_fit = xr.Dataset(
                {
                    'rmm1': (('time',), climate.mjoRmm1),
                    'rmm2': (('time',), climate.mjoRmm2),
                },
                coords={'time': [datetime(climate.mjoYear[r], climate.mjoMonth[r], climate.mjoDay[r]) for r in range(len(climate.mjoDay))]}
                # coords = {'time': timeMJO}
            )
            # reindex to daily data after 1979-01-01 (avoid NaN)
            # self.xds_MJO_fit = xr_daily(self.xds_MJO_fit, datetime(1979, 6, 1), datetime(2024, 5, 31))
            self.xds_MJO_fit = xr_flexible(self.xds_MJO_fit, datetime(1979, 6, 1), datetime(2024, 5, 31),avgTime=self.avgTime)

            # --------------------------------------
            # Mount covariates matrix

            # available data:
            # model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
            # model sim: xds_MJO_sim, xds_PCs_sim

            # covariates: FIT
            # d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_KMA_fit])
            # self.d_covars_fit = xcd_daily([self.xds_PCs_fit, self.xds_MJO_fit, self.xds_KMA_fit])

            self.d_covars_fit = xcd_flexible([self.xds_PCs_fit, self.xds_MJO_fit, self.xds_KMA_fit],avgTime=self.avgTime)

            # PCs covar
            cov_PCs = self.xds_PCs_fit.sel(time=slice(self.d_covars_fit[0], self.d_covars_fit[-1]))
            cov_1 = cov_PCs.PC1.values.reshape(-1, 1)
            cov_2 = cov_PCs.PC2.values.reshape(-1, 1)
            cov_3 = cov_PCs.PC3.values.reshape(-1, 1)

            # MJO covars
            cov_MJO = self.xds_MJO_fit.sel(time=slice(self.d_covars_fit[0], self.d_covars_fit[-1]))
            cov_4 = cov_MJO.rmm1.values.reshape(-1, 1)
            cov_5 = cov_MJO.rmm2.values.reshape(-1, 1)

            # join covars and norm.
            cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

            # generate xarray.Dataset
            cov_names = ['PC1', 'PC2', 'PC3', 'MJO1', 'MJO2']
            self.xds_cov_fit = xr.Dataset(
                {
                    'cov_values': (('time', 'cov_names'), cov_T),
                },
                coords={
                    'time': self.d_covars_fit,
                    'cov_names': cov_names,
                }
            )

            # use bmus inside covariate time frame
            self.xds_bmus_fit = self.xds_KMA_fit.sel(
                time=slice(self.d_covars_fit[0], self.d_covars_fit[-1])
            )

            bmus = self.xds_bmus_fit.bmus

            # Autoregressive logistic wrapper
            num_clusters = int(self.numClusters)#self.num_clustersTC+self.num_clustersETC
            sim_num = 100
            fit_and_save = True  # False for loading
            p_test_ALR = '/Users/dylananderson/Documents/duneLifeCycles/'

            # ALR terms
            self.d_terms_settings = {
                'mk_order': 2,
                'constant': True,
                'long_term': True,
                'seasonality': (True, [2, 4, 6]),
                'covariates': (True, self.xds_cov_fit),
            }
            # Autoregressive logistic wrapper
            ALRW = ALR_WRP(p_test_ALR)
            ALRW.SetFitData(num_clusters, self.xds_bmus_fit, self.d_terms_settings,self.avgTime)

            ALRW.FitModel(max_iter=15000)

            # p_report = op.join(p_data, 'r_{0}'.format(name_test))

            ALRW.Report_Fit()  # '/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)

            if historicalSimNum > 0:
                # Historical Simulation
                # start simulation at PCs available data
                d1 = datetime(1979, 6, 1)  # x2d(xds_cov_fit.time[0])
                d2 = datetime(2024, 5, 31)  # datetime(d1.year+sim_years, d1.month, d1.day)
                # dates_sim = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]
                dates_sim = [d1 + timedelta(hours=int(self.avgTime*i)) for i in range(int((d2 - d1).days*(24/self.avgTime) + (24/self.avgTime)))]

                # print some info
                # print('ALR model fit   : {0} --- {1}'.format(
                #    d_covars_fit[0], d_covars_fit[-1]))
                print('ALR model sim   : {0} --- {1}'.format(
                    dates_sim[0], dates_sim[-1]))

                #  launch simulation
                xds_ALR = ALRW.Simulate(historicalSimNum, dates_sim, self.xds_cov_fit,avgTime=self.avgTime)

                self.historicalDatesSim = dates_sim

                # Save results for matlab plot
                self.historicalBmusSim = xds_ALR.evbmus_sims.values
                # evbmus_probcum = xds_ALR.evbmus_probcum.values


            if futureSimNum > 0:
                self.futureSimStart = futureSimStart
                self.futureSimEnd = futureSimEnd
                futureSims = []
                for simIndex in range(futureSimNum):
                    print('Future Sim #:{}'.format(simIndex))

                    actualSim = np.remainder(simIndex,100)

                    # ALR FUTURE model simulations
                    sim_years = 100
                    # start simulation at PCs available data
                    d1 = datetime(self.futureSimStart, 6, 1)  # x2d(xds_cov_fit.time[0])
                    d2 = datetime(self.futureSimEnd, 5, 31)  # datetime(d1.year+sim_years, d1.month, d1.day)
                    # self.future_dates_sim = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]
                    self.future_dates_sim = [d1 + timedelta(hours=(self.avgTime*i)) for i in
                                 range(int((d2 - d1).days * (24 / self.avgTime) + (24 / self.avgTime)))]

                    d1 = datetime(self.futureSimStart, 6, 1)
                    dt = datetime(self.futureSimStart, 6, 1)
                    end = datetime(self.futureSimEnd, 6, 1)
                    step = relativedelta(years=1)
                    simAnnualTime = []
                    while dt < end:
                        simAnnualTime.append(dt)
                        dt += step

                    d1 = datetime(self.futureSimStart, 6, 1)
                    dt = datetime(self.futureSimStart, 6, 1)
                    end = datetime(self.futureSimEnd, 6, 2)
                    # step = datetime.timedelta(months=1)
                    # step = relativedelta(days=1)
                    step = relativedelta(hours=self.avgTime)

                    simDailyTime = []
                    while dt < end:
                        simDailyTime.append(dt)
                        dt += step
                    simDailyDatesMatrix = np.array([[r.year, r.month, r.day, r.hour] for r in simDailyTime])

                    trainingDates = [datetime(r[0], r[1], r[2], r[3]) for r in simDailyDatesMatrix]
                    dailyAWTsim = np.ones((len(trainingDates),))
                    dailyPC1sim = np.ones((len(trainingDates),))
                    dailyPC2sim = np.ones((len(trainingDates),))
                    dailyPC3sim = np.ones((len(trainingDates),))

                    awtBMUsim = climate.awtBmusSim[actualSim][0:100]  # [0:len(awt_bmus)]
                    awtPC1sim = climate.pc1Sims[actualSim][0:100]  # [0:len(awt_bmus)]
                    awtPC2sim = climate.pc2Sims[actualSim][0:100]  # [0:len(awt_bmus)]
                    awtPC3sim = climate.pc3Sims[actualSim][0:100]  # [0:len(awt_bmus)]
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
                        coords={'time': [datetime(r[0], r[1], r[2],r[3],0,0) for r in simDailyDatesMatrix]}
                    )
                    # reindex annual data to daily data
                    # self.xds_PCs_sim = xr_daily(self.xds_PCs_sim)
                    self.xds_PCs_sim = xr_flexible(self.xds_PCs_sim,avgTime=self.avgTime)


                    # MJO: PCs (Generated with copula simulation. Annual data, parse to daily)
                    self.xds_MJO_sim = xr.Dataset(
                        {
                            'rmm1': (('time',), climate.mjoFutureSimRmm1[actualSim].flatten()),
                            'rmm2': (('time',), climate.mjoFutureSimRmm2[actualSim].flatten()),
                        },
                        coords={'time': [datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix[::2]]}
                    )
                    # reindex annual data to daily data
                    # self.xds_MJO_sim = xr_daily(self.xds_MJO_sim)
                    self.xds_MJO_sim = xr_flexible(self.xds_MJO_sim,avgTime=self.avgTime)



                    # d_covars_sim = xcd_daily([self.xds_PCs_sim,self.xds_MJO_sim])
                    d_covars_sim = xcd_flexible([self.xds_PCs_sim,self.xds_MJO_sim],avgTime=self.avgTime)

                    # PCs covar
                    cov_PCs = self.xds_PCs_sim.sel(time=slice(d_covars_sim[0], d_covars_sim[-1]))
                    cov_1 = cov_PCs.PC1.values.reshape(-1, 1)
                    cov_2 = cov_PCs.PC2.values.reshape(-1, 1)
                    cov_3 = cov_PCs.PC3.values.reshape(-1, 1)

                    # MJO covars
                    cov_MJO = self.xds_MJO_sim.sel(time=slice(d_covars_sim[0], d_covars_sim[-1]))
                    cov_4 = cov_MJO.rmm1.values.reshape(-1, 1)
                    cov_5 = cov_MJO.rmm2.values.reshape(-1, 1)

                    # join covars and norm.
                    cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

                    # generate xarray.Dataset
                    cov_names = ['PC1', 'PC2', 'PC3', 'MJO1', 'MJO2']
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
                        1, self.future_dates_sim, self.xds_cov_sim,avgTime=self.avgTime)

                    self.futureDatesSim = self.future_dates_sim

                    futureSims.append(xds_ALRfuture.evbmus_sims.values)


                # Save results for matlab plot
                self.futureBmusSim = futureSims
                # evbmus_probcum = xds_ALR.evbmus_probcum.values
                # convert synthetic markovs to PC values





                import itertools
                import operator
                from datetime import timedelta
                import random

                dt = datetime(1979, 6, 1)
                end = datetime(2024, 6, 1)
                # step = timedelta(days=1)
                step = timedelta(hours=self.avgTime)

                midnightTime = []
                while dt < end:
                    midnightTime.append(dt)  # .strftime('%Y-%m-%d'))
                    dt += step
                # Import the time library
                import time


                groupedList = list()
                groupLengthList = list()
                bmuGroupList = list()
                # timeGroupList = list()
                for hh in range(historicalSimNum):
                    # Calculate the start time
                    start = time.time()
                    print('breaking up hydrogrpahs for historical sim {}'.format(hh))
                    bmusTemp = self.historicalBmusSim[:,hh]
                    tempBmusGroup = [[e[0] for e in d[1]] for d in
                                     itertools.groupby(enumerate(bmusTemp), key=operator.itemgetter(1))]
                    groupedList.append(tempBmusGroup)
                    groupLengthList.append(np.asarray([len(i) for i in tempBmusGroup]))
                    bmuGroupList.append(np.asarray([bmusTemp[i[0]] for i in tempBmusGroup]))
                    # timeGroupList.append([np.asarray(midnightTime)[i] for i in tempBmusGroup]) # THIS SLOWS EVERYTHING DOWN 100 FOLD
                    # Calculate the end time and time taken
                    end = time.time()
                    length = end - start
                    print('completed in {} minutes'.format(length/60))

                simBmuChopped = []
                simBmuLengthChopped = []
                simBmuGroupsChopped = []
                for pp in range(historicalSimNum):
                    start = time.time()

                    print('working on historical realization #{}'.format(pp))
                    bmuGroup = bmuGroupList[pp]
                    groupLength = groupLengthList[pp]
                    grouped = groupedList[pp]
                    simGroupLength = []
                    simGrouped = []
                    simBmu = []
                    for i in range(len(groupLength)):
                        # if np.remainder(i,10000) == 0:
                        #     print('done with {} hydrographs'.format(i))
                        tempGrouped = grouped[i]
                        tempBmu = int(bmuGroup[i])
                        remainingDays = groupLength[i] - int(5*(24/self.avgTime))
                        if groupLength[i] < int(5*(24/self.avgTime)):
                            simGroupLength.append(int(groupLength[i]))
                            simGrouped.append(grouped[i])
                            simBmu.append(tempBmu)
                        else:
                            counter = 0
                            while (len(grouped[i]) - counter) > int(5*(24/self.avgTime)):
                                # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                                # random days between 3 and 5
                                randLength = random.randint(1, int(3*(24/self.avgTime))) + int(2*(24/self.avgTime))
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
                    # Calculate the end time and time taken
                    end = time.time()
                    length = end - start
                    print('completed in {} minutes'.format(length/60))

                self.simHistBmuLengthChopped = simBmuLengthChopped
                self.simHistBmuGroupsChopped = simBmuGroupsChopped
                self.simHistBmuChopped = simBmuChopped


                dt = datetime(self.futureSimStart, 6, 1)
                end = datetime(self.futureSimEnd, 6, 2)
                # step = timedelta(days=1)
                step = timedelta(hours=self.avgTime)

                midnightTime = []
                while dt < end:
                    midnightTime.append(dt)  # .strftime('%Y-%m-%d'))
                    dt += step

                groupedListFuture = list()
                groupLengthListFuture = list()
                bmuGroupListFuture = list()
                # timeGroupListFuture = list()
                for hh in range(futureSimNum):
                    start = time.time()

                    print('breaking up hydrogrpahs for future sim {}'.format(hh))
                    bmusTemp = self.futureBmusSim[hh].flatten()
                    tempBmusGroup = [[e[0] for e in d[1]] for d in
                                     itertools.groupby(enumerate(bmusTemp), key=operator.itemgetter(1))]
                    groupedListFuture.append(tempBmusGroup)
                    groupLengthListFuture.append(np.asarray([len(i) for i in tempBmusGroup]))
                    bmuGroupListFuture.append(np.asarray([bmusTemp[i[0]] for i in tempBmusGroup]))
                    # timeGroupListFuture.append([np.asarray(midnightTime)[i] for i in tempBmusGroup]) # THIS SLOWS EVERYTHING DOWN 100 FOLD
                    # Calculate the end time and time taken
                    end = time.time()
                    length = end - start
                    print('completed in {} minutes'.format(length/60))

                simBmuChopped = []
                simBmuLengthChopped = []
                simBmuGroupsChopped = []
                for pp in range(futureSimNum):
                    start = time.time()
                    print('working on future realization #{}'.format(pp))
                    bmuGroupFuture = bmuGroupListFuture[pp]
                    groupLengthFuture = groupLengthListFuture[pp]
                    groupedFuture = groupedListFuture[pp]
                    simGroupLength = []
                    simGrouped = []
                    simBmu = []
                    for i in range(len(groupLengthFuture)):
                        # if np.remainder(i,10000) == 0:
                        #     print('done with {} hydrographs'.format(i))
                        tempGrouped = groupedFuture[i]
                        tempBmu = int(bmuGroupFuture[i])
                        remainingDays = groupLengthFuture[i] - int(5*(24/self.avgTime))
                        if groupLengthFuture[i] < int(5*(24/self.avgTime)):
                            simGroupLength.append(int(groupLengthFuture[i]))
                            simGrouped.append(groupedFuture[i])
                            simBmu.append(tempBmu)
                        else:
                            counter = 0
                            while (len(groupedFuture[i]) - counter) > int(5*(24/self.avgTime)):
                                # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                                # random days between 3 and 5
                                randLength = random.randint(1, int(3*(24/self.avgTime))) + int(2*(24/self.avgTime))
                                # add this to the record
                                simGroupLength.append(int(randLength))
                                # simGrouped.append(tempGrouped[0:randLength])
                                # print('should be adding {}'.format(grouped[i][counter:counter+randLength]))
                                simGrouped.append(groupedFuture[i][counter:counter + randLength])
                                simBmu.append(tempBmu)
                                # remove those from the next step
                                # tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))
                                # do we continue forward
                                remainingDays = remainingDays - randLength
                                counter = counter + randLength

                            if (len(groupedFuture[i]) - counter) > 0:
                                simGroupLength.append(int((len(groupedFuture[i]) - counter)))
                                # simGrouped.append(tempGrouped[0:])
                                simGrouped.append(groupedFuture[i][counter:])
                                simBmu.append(tempBmu)
                    simBmuLengthChopped.append(np.asarray(simGroupLength))
                    simBmuGroupsChopped.append(simGrouped)
                    simBmuChopped.append(np.asarray(simBmu))
                    # Calculate the end time and time taken
                    end = time.time()
                    length = end - start
                    print('completed in {} minutes'.format(length/60))
                self.simFutureBmuLengthChopped = simBmuLengthChopped
                self.simFutureBmuGroupsChopped = simBmuGroupsChopped
                self.simFutureBmuChopped = simBmuChopped




                import pickle
                samplesPickle = 'simDwts.pickle'
                outputSamples = {}
                outputSamples['futureBmusSim'] = self.futureBmusSim
                outputSamples['futureDatesSim'] = self.futureDatesSim
                outputSamples['historicalDatesSim'] = self.historicalDatesSim
                outputSamples['historicalBmusSim'] = self.historicalBmusSim
                outputSamples['simHistBmuLengthChopped'] = self.simHistBmuLengthChopped
                outputSamples['simHistBmuGroupsChopped'] = self.simHistBmuGroupsChopped
                outputSamples['simHistBmuChopped'] = self.simHistBmuChopped
                outputSamples['simFutureBmuLengthChopped'] = self.simFutureBmuLengthChopped
                outputSamples['simFutureBmuGroupsChopped'] = self.simFutureBmuGroupsChopped
                outputSamples['simFutureBmuChopped'] = self.simFutureBmuChopped
                outputSamples['futureSimEnd'] = self.futureSimEnd
                outputSamples['futureSimStart'] = self.futureSimStart

                with open(os.path.join(self.savePath,samplesPickle), 'wb') as f:
                    pickle.dump(outputSamples, f)

    def separateHistoricalHydrographs(self,metOcean,numRealizations=100,loadPrior = False,loadPickle = './'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                hyds = pickle.load(input_file)

            self.hydros = hyds['hydros']
            self.bmuGroup = hyds['bmuGroup']
            self.groupLength = hyds['groupLength']
            self.grouped = hyds['grouped']
            self.histBmuLengthChopped = hyds['histBmuLengthChopped']
            self.histBmuGroupsChopped = hyds['histBmuGroupsChopped']
            self.histBmuChopped = hyds['histBmuChopped']

            print('loaded prior hydros')

        else:

            import itertools
            import operator
            from datetime import timedelta
            import random
            import numpy as np

            bmus = self.bmus_corrected
            time = self.DATES

            dt = time[0]
            end = time[-1]
            step = timedelta(hours=self.avgTime)
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
            midnightTimeArray = np.asarray(midnightTime)
            timeGroupList.append([midnightTimeArray[i] for i in tempBmusGroup]) # THIS SLOWS EVERYTHING DOWN 100 FOLD

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
                    remainingDays = groupLength[i] - (5*24/self.avgTime)
                    if groupLength[i] < (5*24/self.avgTime):
                        simGroupLength.append(int(groupLength[i]))
                        simGrouped.append(grouped[i])
                        simBmu.append(tempBmu)
                    else:
                        counter = 0
                        while (len(grouped[i]) - counter) > (5*24/self.avgTime):
                            # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                            # random days between 3 and 5
                            randLength = random.randint(1, (3*24/self.avgTime)) + int((2*24/self.avgTime))
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

            self.histBmuLengthChopped = simBmuLengthChopped
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
            ntr = metOcean.resWl#[beginTime[0][0]:endingTime[0][0] + 24]

            # waveNorm = dm - metOcean.shoreNormal
            # neg = np.where((waveNorm > 180))
            # waveNorm[neg[0]] = waveNorm[neg[0]] - 360
            # neg2 = np.where((waveNorm < -180))
            # waveNorm[neg2[0]] = waveNorm[neg2[0]] + 360
            # dmOG = dm
            # dm = waveNorm

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
                    et = endTimes[index[i]] + timedelta(hours=self.avgTime)

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
                    else:
                        print('we couldnt add data on {} because its missing waves or water levels'.format(st))

                print('we collected {} of {} hydrographs due to data gaps in weather pattern {}'.format(counter, len(index), p))
                hydros.append(tempList)
            self.hydros = hydros
            self.bmuGroup = bmuGroupList[0]
            self.groupLength = groupLengthList[0]
            self.grouped = groupedList[0]



            import pickle
            samplesPickle = 'hydros.pickle'
            outputSamples = {}
            outputSamples['hydros'] = self.hydros
            outputSamples['bmuGroup'] = self.bmuGroup
            outputSamples['groupLength'] = self.groupLength
            outputSamples['grouped'] = self.grouped
            outputSamples['histBmuLengthChopped'] = self.histBmuLengthChopped
            outputSamples['histBmuGroupsChopped'] = self.histBmuGroupsChopped
            outputSamples['histBmuChopped'] = self.histBmuChopped

            with open(os.path.join(self.savePath, samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)

    def separateHistoricalHydrographsWinds(self, metOcean, numRealizations=100, loadPrior=False, loadPickle='./'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                hyds = pickle.load(input_file)

            self.hydros = hyds['hydros']
            self.bmuGroup = hyds['bmuGroup']
            self.groupLength = hyds['groupLength']
            self.grouped = hyds['grouped']
            self.histBmuLengthChopped = hyds['histBmuLengthChopped']
            self.histBmuGroupsChopped = hyds['histBmuGroupsChopped']
            self.histBmuChopped = hyds['histBmuChopped']

            print('loaded prior hydros')

        else:

            import itertools
            import operator
            from datetime import timedelta
            import random
            import numpy as np

            bmus = self.bmus_corrected
            time = self.DATES

            dt = time[0]
            end = time[-1]
            step = timedelta(hours=self.avgTime)
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
            midnightTimeArray = np.asarray(midnightTime)
            timeGroupList.append(
                [midnightTimeArray[i] for i in tempBmusGroup])  # THIS SLOWS EVERYTHING DOWN 100 FOLD

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
                    remainingDays = groupLength[i] - (5 * 24 / self.avgTime)
                    if groupLength[i] < (5 * 24 / self.avgTime):
                        simGroupLength.append(int(groupLength[i]))
                        simGrouped.append(grouped[i])
                        simBmu.append(tempBmu)
                    else:
                        counter = 0
                        while (len(grouped[i]) - counter) > (5 * 24 / self.avgTime):
                            # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                            # random days between 3 and 5
                            randLength = random.randint(1, (3 * 24 / self.avgTime)) + int(
                                (2 * 24 / self.avgTime))
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

            self.histBmuLengthChopped = simBmuLengthChopped
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

            wh = metOcean.Hs  # [beginTime[0][0]:endingTime[0][0] + 24]
            tp = metOcean.Tp  # [beginTime[0][0]:endingTime[0][0] + 24]
            dm = metOcean.Dm  # [beginTime[0][0]:endingTime[0][0] + 24]
            ntr = metOcean.resWl  # [beginTime[0][0]:endingTime[0][0] + 24]
            u10 = metOcean.u10
            v10 = metOcean.v10

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
                    et = endTimes[index[i]] + timedelta(hours=self.avgTime)

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
                        tempDict['v10'] = v10[waveInd[0]]
                        tempDict['u10'] = u10[waveInd[0]]
                        tempDict['cop'] = np.asarray([np.nanmin(wh[waveInd[0]]), np.nanmax(wh[waveInd[0]]),
                                                      np.nanmin(tp[waveInd[0]]), np.nanmax(tp[waveInd[0]]),
                                                      np.nanmin(u10[waveInd[0]]), np.nanmax(u10[waveInd[0]]),
                                                      np.nanmin(v10[waveInd[0]]), np.nanmax(v10[waveInd[0]]),
                                                      np.nanmean(dm[waveInd[0]]),
                                                      np.nanmean(ntr[ntrInd[0]])])

                        tempDict['hsMin'] = np.nanmin(wh[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(wh[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tp[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tp[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(dm[waveInd[0]])
                        tempDict['uMin'] = np.nanmin(u10[waveInd[0]])
                        tempDict['uMax'] = np.nanmax(u10[waveInd[0]])
                        tempDict['vMin'] = np.nanmin(v10[waveInd[0]])
                        tempDict['vMax'] = np.nanmax(v10[waveInd[0]])
                        tempDict['ntrMean'] = np.nanmean(ntr[ntrInd[0]])
                        tempDict['ntrMin'] = np.nanmin(ntr[ntrInd[0]])
                        tempDict['ntrMax'] = np.nanmax(ntr[ntrInd[0]])
                        tempList.append(tempDict)
                        tempList.append(tempDict)
                    else:
                        print('we couldnt add data on {} because its missing waves or water levels'.format(st))

                print('we collected {} of {} hydrographs due to data gaps in weather pattern {}'.format(counter, len(index), p))
                hydros.append(tempList)
            self.hydros = hydros
            self.bmuGroup = bmuGroupList[0]
            self.groupLength = groupLengthList[0]
            self.grouped = groupedList[0]

            import pickle
            samplesPickle = 'hydros.pickle'
            outputSamples = {}
            outputSamples['hydros'] = self.hydros
            outputSamples['bmuGroup'] = self.bmuGroup
            outputSamples['groupLength'] = self.groupLength
            outputSamples['grouped'] = self.grouped
            outputSamples['histBmuLengthChopped'] = self.histBmuLengthChopped
            outputSamples['histBmuGroupsChopped'] = self.histBmuGroupsChopped
            outputSamples['histBmuChopped'] = self.histBmuChopped

            with open(os.path.join(self.savePath, samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)

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



    def metOceanCopulas(self,loadPrior = False,loadPickle = './'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                cops = pickle.load(input_file)

            self.gevCopulaSims = cops['gevCopulaSims']
            self.normalizedHydros = cops['normalizedHydros']
            self.bmuDataMin = cops['bmuDataMin']
            self.bmuDataMax = cops['bmuDataMax']
            self.bmuDataStd = cops['bmuDataStd']
            self.bmuDataNormalized = cops['bmuDataNormalized']
            self.copulaData = cops['copulaData']

            print('loaded prior copulas')

        else:






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
                        print('woah, no waves here')
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
                    # Hsmax, Hsmin, Tpmax, Tpmin, Dmmean, NTRmean, time, kk

                    data2 = [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
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
            self.gevCopulaSims = gevCopulaSims



            import pickle
            samplesPickle = 'copulas.pickle'
            outputSamples = {}
            outputSamples['gevCopulaSims'] = self.gevCopulaSims
            outputSamples['normalizedHydros'] = self.normalizedHydros
            outputSamples['bmuDataMin'] = self.bmuDataMin
            outputSamples['bmuDataMax'] = self.bmuDataMax
            outputSamples['bmuDataStd'] = self.bmuDataStd
            outputSamples['bmuDataNormalized'] = self.bmuDataNormalized
            outputSamples['copulaData'] = self.copulaData


            with open(os.path.join(self.savePath, samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)



    def metOceanCopulasWinds(self,loadPrior = False,loadPickle = './'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                cops = pickle.load(input_file)

            self.gevCopulaSims = cops['gevCopulaSims']
            self.normalizedHydros = cops['normalizedHydros']
            self.bmuDataMin = cops['bmuDataMin']
            self.bmuDataMax = cops['bmuDataMax']
            self.bmuDataStd = cops['bmuDataStd']
            self.bmuDataNormalized = cops['bmuDataNormalized']
            self.copulaData = cops['copulaData']

            print('loaded prior copulas')

        else:






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
                                         tempHydros[kk]['tpMin'], tempHydros[kk]['uMax'], tempHydros[kk]['uMin'],
                                         tempHydros[kk]['vMax'], tempHydros[kk]['vMin'],
                                         tempHydros[kk]['dmMean'], tempHydros[kk]['ntrMean'], len(tempHydros[kk]['time']), kk]))

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
                        print('woah, no waves here')
                        bmuDataNormalized.append(np.vstack((0, 0)).T)
                        bmuDataMin.append([0, 0])
                        bmuDataMax.append([0, 0])
                        bmuDataStd.append([0, 0])
                    else:

                        maxDm = np.nanmax(data2[:, 8])
                        minDm = np.nanmin(data2[:, 8])
                        stdDm = np.nanstd(data2[:, 8])
                        dmNorm = (data2[:, 8] - minDm) / (maxDm - minDm)
                        maxSs = np.nanmax(data2[:, 9])
                        minSs = np.nanmin(data2[:, 9])
                        stdSs = np.nanstd(data2[:, 9])
                        ssNorm = (data2[:, 9] - minSs) / (maxSs - minSs)
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
                        tempDict['uNorm'] = (tempHydros[mm]['u10'] - tempHydros[mm]['uMin']) / (
                                tempHydros[mm]['uMax'] - tempHydros[mm]['uMin'])
                        tempDict['vNorm'] = (tempHydros[mm]['v10'] - tempHydros[mm]['vMin']) / (
                                tempHydros[mm]['vMax'] - tempHydros[mm]['vMin'])
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
                    # Hsmax, Hsmin, Tpmax, Tpmin, Dmmean, NTRmean, time, kk

                    data2 = [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
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
                    kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE']
                elif len(data2) == 3 or len(data2) == 2:
                    kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE']
                    data2 = np.vstack((data2, data2 - data2 * 0.1))
                else:
                    kernels = ['KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE', 'KDE']

                if len(data2) <= 1:
                    samples5 = np.zeros((100000, 10))
                else:
                    samples = copulaSimulation(data2[:, 0:10], kernels, 100000)

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
            self.gevCopulaSims = gevCopulaSims



            import pickle
            samplesPickle = 'copulas.pickle'
            outputSamples = {}
            outputSamples['gevCopulaSims'] = self.gevCopulaSims
            outputSamples['normalizedHydros'] = self.normalizedHydros
            outputSamples['bmuDataMin'] = self.bmuDataMin
            outputSamples['bmuDataMax'] = self.bmuDataMax
            outputSamples['bmuDataStd'] = self.bmuDataStd
            outputSamples['bmuDataNormalized'] = self.bmuDataNormalized
            outputSamples['copulaData'] = self.copulaData


            with open(os.path.join(self.savePath, samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)


    def simsFutureInterpolated(self,simNum,nodePath):
        import numpy as np
        from datetime import datetime, date, timedelta
        import random
        from scipy.spatial import distance
        import pickle
        import calendar
        import pandas
        dt = datetime(self.futureSimStart, 6, 1, 0, 0, 0)
        end = datetime(self.futureSimEnd, 5, 31, 23, 0, 0)
        step = timedelta(hours=1)
        hourlyTime = []
        while dt < end:
            hourlyTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        deltaT = [(tt - hourlyTime[0]).total_seconds() / (3600 * 24) for tt in hourlyTime]

        def closest_node(node, nodes):
            closest_index = distance.cdist([node], nodes).argmin()
            return nodes[closest_index], closest_index

        simulationsHs = list()
        simulationsTp = list()
        simulationsDm = list()
        simulationsSs = list()
        simulationsTime = list()
        simulationsU10 = list()
        simulationsV10 = list()
        # plus = 900

        for simNum in range(simNum):
            simHs = []
            simTp = []
            simDm = []
            simSs = []
            simTime = []
            simWLTime = []
            simU10 = []
            simV10 = []
            print('filling in simulation #{}'.format(simNum))

            for i in range(len(self.simFutureBmuChopped[simNum])):
                if np.remainder(i, 1000) == 0:
                    print('done with {} hydrographs'.format(i))
                tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                randStorm = random.randint(0, 9999)
                stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                if stormDetails[0] > 8:
                    print('oh boy, we''ve picked a {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                    tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                    randStorm = random.randint(0, 9999)
                    stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                # if stormDetails[0] > 1000:
                #     print('ok, this is where we''ve picked a {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                #     tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                #     randStorm = random.randint(0, 9999)
                #     stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #     stormDets1 = stormDetails
                #     if stormDetails[6] > 55:
                #         print('oh boy, we''ve picked a {}m/s storm wave in BMU #{} again!'.format(stormDetails[6],
                #                                                                                   tempBmu))
                #         tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                #         randStorm = random.randint(0, 9999)
                #         stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #         stormDets2 = stormDetails
                #         if stormDetails[6] > 55:
                #             print('oh boy, we''ve picked a {}m/s storm wave in BMU #{} a third time!'.format(
                #                 stormDetails[6], tempBmu))
                #             stormDetails[6] = 55
                #             stormDetails[7] = 25

                durSim = self.simFutureBmuLengthChopped[simNum][i]

                simDmNorm = (stormDetails[4] - np.asarray(self.bmuDataMin)[tempBmu, 0]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 0] - np.asarray(self.bmuDataMin)[tempBmu, 0])
                simSsNorm = (stormDetails[5] - np.asarray(self.bmuDataMin)[tempBmu, 1]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 1] - np.asarray(self.bmuDataMin)[tempBmu, 1])
                test, closeIndex = closest_node([simDmNorm, simSsNorm], np.asarray(self.bmuDataNormalized)[tempBmu])
                actualIndex = int(np.asarray(self.copulaData[tempBmu])[closeIndex, 7])

                tempHs = ((self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0] - stormDetails[1]) +
                          stormDetails[1]).filled()
                tempTp = ((self.normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2] - stormDetails[3]) +
                          stormDetails[3]).filled()
                # tempWind = ((self.normalizedHydros[tempBmu][actualIndex]['wNorm']) * (stormDetails[6] - stormDetails[7]) +
                #             stormDetails[7]).filled()
                # tempWindDir = ((self.normalizedHydros[tempBmu][actualIndex]['wdNorm']) + stormDetails[8])

                tempDm = ((self.normalizedHydros[tempBmu][actualIndex]['dmNorm']) + stormDetails[4])
                tempSs = ((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])

                tempWLtime = np.arange(0,durSim,durSim/len(tempSs))

                if len(self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(
                        self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu, actualIndex))
                if stormDetails[1] < 0:
                    print('woah, we''re less than 0 over here')
                    asdfg
                # if len(tempSs) < len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                #     # print('Ss is shorter than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                #     tempLength = len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'])
                #     tempSs = np.zeros((len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']),))
                #     tempSs[0:len((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])] = (
                #                 (self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])

                # if len(tempSs) > len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    # print('Now Ss is longer than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                    # print('{} vs. {}'.format(len(tempSs),len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])))
                    # tempSs = tempSs[0:-1]
                if len(tempSs) > len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is longer than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempSs = tempSs[0:-1]
                if len(tempSs) < len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is shorter than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempWLtime = tempWLtime[0:-1]


                simHs.append(tempHs)
                simTp.append(tempTp)
                simDm.append(tempDm)
                simSs.append(tempSs)
                # simWind.append(tempWind)
                # simWindDir.append(tempWindDir)
                # simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                # dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                simTime.append(np.hstack((np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim),
                                          np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim)[-1])))
                simWLTime.append(np.hstack((np.diff(tempWLtime),np.diff(tempWLtime)[-1])))



            cumulativeHours = np.cumsum(np.hstack(simTime))
            # newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
            newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeHours]

            simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyTime]

            # Just for water levels at different time interval
            cumulativeWLHours = np.cumsum(np.hstack(simWLTime))
            newDailyWLTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeWLHours]
            simDeltaWLT = [(tt - newDailyWLTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyWLTime]

            print('water level time vs. surge: {} vs {}'.format(len(np.hstack(simSs)),len(simDeltaWLT)))

            # simData = np.array(
            #     np.vstack((np.hstack(simHs).T, np.hstack(simTp).T, np.hstack(simDm).T, np.hstack(simSs).T)))
            # # simData = np.array((np.ma.asarray(np.hstack(simHs)),np.ma.asarray(np.hstack(simTp)),np.ma.asarray(np.hstack(simDm)),np.ma.asarray(np.hstack(simSs))))
            # # simData = np.array([np.hstack(simHs).filled(),np.hstack(simTp).filled(),np.hstack(simDm).filled(),np.hstack(simSs)])
            #
            # ogdf = pandas.DataFrame(data=simData.T, index=newDailyTime, columns=["hs", "tp", "dm", "ss"])

            print('interpolating')
            interpHs = np.interp(deltaT, simDeltaT, np.hstack(simHs))
            interpTp = np.interp(deltaT, simDeltaT, np.hstack(simTp))
            interpDm = np.interp(deltaT, simDeltaT, np.hstack(simDm))
            # interpWind = np.interp(deltaT, simDeltaT, np.hstack(simWind))
            # interpWindDir = np.interp(deltaT, simDeltaT, np.hstack(simWindDir))
            interpSs = np.interp(deltaT, simDeltaWLT, np.hstack(simSs))

            badWaves = np.where(interpHs > 10)
            interpHs[badWaves] = interpHs[badWaves]*0+1.5

            simDataInterp = np.array([interpHs, interpTp, interpDm, interpSs])#, interpWind, interpWindDir])

            df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "ss"])
            # df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "ss", "w", "wd"])

            # resampled = df.resample('H')
            # interped = resampled.interpolate()
            # simulationData = interped.values
            # testTime = interped.index  # to_pydatetime()
            # testTime2 = testTime.to_pydatetime()

            # simsPickle = ('/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle'.format(simNum))
            # simsPickle = ('/media/dylananderson/Elements/Sims/simulation{}.pickle'.format(simNum))
            simsPickle = ('futureSims{}.pickle'.format(simNum))

            outputSims = {}
            outputSims['simulationData'] = simDataInterp.T
            outputSims['df'] = df
            outputSims['simHs'] = np.hstack(simHs)
            outputSims['simTp'] = np.hstack(simTp)
            outputSims['simDm'] = np.hstack(simDm)
            outputSims['simSs'] = np.hstack(simSs)
            # outputSims['simWind'] = np.hstack(simWind)
            # outputSims['simWindDir'] = np.hstack(simWindDir)

            outputSims['time'] = hourlyTime

            with open(os.path.join(nodePath,simsPickle), 'wb') as f:
                pickle.dump(outputSims, f)


    def simsFutureInterpolatedWinds(self,simNum,nodePath):
        import numpy as np
        from datetime import datetime, date, timedelta
        import random
        from scipy.spatial import distance
        import pickle
        import calendar
        import pandas
        dt = datetime(self.futureSimStart, 6, 1, 0, 0, 0)
        end = datetime(self.futureSimEnd, 5, 31, 23, 0, 0)
        step = timedelta(hours=1)
        hourlyTime = []
        while dt < end:
            hourlyTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        deltaT = [(tt - hourlyTime[0]).total_seconds() / (3600 * 24) for tt in hourlyTime]

        def closest_node(node, nodes):
            closest_index = distance.cdist([node], nodes).argmin()
            return nodes[closest_index], closest_index

        simulationsHs = list()
        simulationsTp = list()
        simulationsDm = list()
        simulationsSs = list()
        simulationsTime = list()
        simulationsU10 = list()
        simulationsV10 = list()
        # plus = 900

        for simNum in range(simNum):
            simHs = []
            simTp = []
            simDm = []
            simSs = []
            simTime = []
            simWLTime = []
            simU10 = []
            simV10 = []
            print('filling in simulation #{}'.format(simNum))

            for i in range(len(self.simFutureBmuChopped[simNum])):
                if np.remainder(i, 1000) == 0:
                    print('done with {} hydrographs'.format(i))
                tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                randStorm = random.randint(0, 80000)
                stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                if stormDetails[0] > 8:
                    print('oh boy, we''ve picked a {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                    tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                    randStorm = random.randint(0, 80000)
                    stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                if stormDetails[0] > 8:
                    print('yikes, we picked another {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                    index = np.where((self.gevCopulaSims[tempBmu][:, 0] < 2.5))
                    subsetGEV = self.gevCopulaSims[tempBmu][index[0], :]
                    randStorm = random.randint(0, len(index))
                    stormDetails = subsetGEV[randStorm]

                # if stormDetails[0] > 1000:
                #     print('ok, this is where we''ve picked a {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                #     tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                #     randStorm = random.randint(0, 9999)
                #     stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #     stormDets1 = stormDetails
                #     if stormDetails[6] > 55:
                #         print('oh boy, we''ve picked a {}m/s storm wave in BMU #{} again!'.format(stormDetails[6],
                #                                                                                   tempBmu))
                #         tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                #         randStorm = random.randint(0, 9999)
                #         stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #         stormDets2 = stormDetails
                #         if stormDetails[6] > 55:
                #             print('oh boy, we''ve picked a {}m/s storm wave in BMU #{} a third time!'.format(
                #                 stormDetails[6], tempBmu))
                #             stormDetails[6] = 55
                #             stormDetails[7] = 25

                durSim = self.simFutureBmuLengthChopped[simNum][i]/(24/self.avgTime)

                simDmNorm = (stormDetails[8] - np.asarray(self.bmuDataMin)[tempBmu, 0]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 0] - np.asarray(self.bmuDataMin)[tempBmu, 0])
                simSsNorm = (stormDetails[9] - np.asarray(self.bmuDataMin)[tempBmu, 1]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 1] - np.asarray(self.bmuDataMin)[tempBmu, 1])
                test, closeIndex = closest_node([simDmNorm, simSsNorm], np.asarray(self.bmuDataNormalized)[tempBmu])
                actualIndex = int(np.asarray(self.copulaData[tempBmu])[closeIndex, 11])

                tempHs = ((self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0] - stormDetails[1]) +
                          stormDetails[1]).filled()
                tempTp = ((self.normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2] - stormDetails[3]) +
                          stormDetails[3]).filled()
                tempU = ((self.normalizedHydros[tempBmu][actualIndex]['uNorm']) * (stormDetails[4] - stormDetails[5]) +
                            stormDetails[5])
                tempV = ((self.normalizedHydros[tempBmu][actualIndex]['vNorm']) * (stormDetails[6] - stormDetails[7]) +
                            stormDetails[7])

                tempDm = ((self.normalizedHydros[tempBmu][actualIndex]['dmNorm']) + stormDetails[8])
                tempSs = ((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[9])

                tempWLtime = np.arange(0,durSim,durSim/len(tempSs))

                if len(self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(
                        self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu, actualIndex))
                if stormDetails[1] < 0:
                    print('woah, we''re less than 0 over here')
                    asdfg
                # if len(tempSs) < len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                #     # print('Ss is shorter than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                #     tempLength = len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'])
                #     tempSs = np.zeros((len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']),))
                #     tempSs[0:len((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])] = (
                #                 (self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])

                # if len(tempSs) > len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    # print('Now Ss is longer than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                    # print('{} vs. {}'.format(len(tempSs),len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])))
                    # tempSs = tempSs[0:-1]
                if len(tempSs) > len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is longer than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempSs = tempSs[0:-1]
                if len(tempSs) < len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is shorter than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempWLtime = tempWLtime[0:-1]


                simHs.append(tempHs)
                simTp.append(tempTp)
                simDm.append(tempDm)
                simSs.append(tempSs)
                simU10.append(tempU)
                simV10.append(tempV)
                # simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                # dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                simTime.append(np.hstack((np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim),
                                          np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim)[-1])))
                # print('durSim = {}, len(tempSs) = {}, tempWLtime = {}'.format(durSim,len(tempSs),tempWLtime))

                if len(tempSs)>1:
                    simWLTime.append(np.hstack((np.diff(tempWLtime),np.diff(tempWLtime)[-1])))
                else:
                    #print(tempSs)
                    simWLTime.append(durSim)



            cumulativeHours = np.cumsum(np.hstack(simTime))
            newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
            # newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeHours]

            # simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyTime]
            simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600*24) for tt in newDailyTime]

            # Just for water levels at different time interval
            cumulativeWLHours = np.cumsum(np.hstack(simWLTime))
            # newDailyWLTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeWLHours]
            # simDeltaWLT = [(tt - newDailyWLTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyWLTime]
            newDailyWLTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii) for ii in cumulativeWLHours]
            simDeltaWLT = [(tt - newDailyWLTime[0]).total_seconds() / (3600*24) for tt in newDailyWLTime]

            print('water level time vs. surge: {} vs {}'.format(len(np.hstack(simSs)),len(simDeltaWLT)))

            # simData = np.array(
            #     np.vstack((np.hstack(simHs).T, np.hstack(simTp).T, np.hstack(simDm).T, np.hstack(simSs).T)))
            # # simData = np.array((np.ma.asarray(np.hstack(simHs)),np.ma.asarray(np.hstack(simTp)),np.ma.asarray(np.hstack(simDm)),np.ma.asarray(np.hstack(simSs))))
            # # simData = np.array([np.hstack(simHs).filled(),np.hstack(simTp).filled(),np.hstack(simDm).filled(),np.hstack(simSs)])
            #
            # ogdf = pandas.DataFrame(data=simData.T, index=newDailyTime, columns=["hs", "tp", "dm", "ss"])

            print('interpolating')
            interpHs = np.interp(deltaT, simDeltaT, np.hstack(simHs))
            interpTp = np.interp(deltaT, simDeltaT, np.hstack(simTp))
            interpDm = np.interp(deltaT, simDeltaT, np.hstack(simDm))
            interpU10 = np.interp(deltaT, simDeltaT, np.hstack(simU10))
            interpV10 = np.interp(deltaT, simDeltaT, np.hstack(simV10))
            interpSs = np.interp(deltaT, simDeltaWLT, np.hstack(simSs))

            # badWaves = np.where(interpHs > 10)
            # interpHs[badWaves] = interpHs[badWaves]*0+1.5

            simDataInterp = np.array([interpHs, interpTp, interpDm, interpU10, interpV10,interpSs])#, interpWind, interpWindDir])

            df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "u10","v10","ss"])
            # df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "ss", "w", "wd"])

            # resampled = df.resample('H')
            # interped = resampled.interpolate()
            # simulationData = interped.values
            # testTime = interped.index  # to_pydatetime()
            # testTime2 = testTime.to_pydatetime()

            # simsPickle = ('/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle'.format(simNum))
            # simsPickle = ('/media/dylananderson/Elements/Sims/simulation{}.pickle'.format(simNum))
            simsPickle = ('futureSims{}.pickle'.format(simNum))

            outputSims = {}
            outputSims['simulationData'] = simDataInterp.T
            outputSims['df'] = df
            outputSims['simHs'] = np.hstack(simHs)
            outputSims['simTp'] = np.hstack(simTp)
            outputSims['simDm'] = np.hstack(simDm)
            outputSims['simSs'] = np.hstack(simSs)
            outputSims['simU10'] = np.hstack(simU10)
            outputSims['simV10'] = np.hstack(simV10)

            # outputSims['simWind'] = np.hstack(simWind)
            # outputSims['simWindDir'] = np.hstack(simWindDir)

            outputSims['time'] = hourlyTime

            with open(os.path.join(nodePath,simsPickle), 'wb') as f:
                pickle.dump(outputSims, f)



    def simsHistoricalInterpolated(self,simNum,nodePath):
        import numpy as np
        from datetime import datetime, date, timedelta
        import random
        from scipy.spatial import distance
        import pickle
        import calendar
        import pandas
        dt = datetime(1979, 6, 1, 0, 0, 0)
        end = datetime(2024, 5, 31, 23, 0, 0)
        step = timedelta(hours=1)
        hourlyTime = []
        while dt < end:
            hourlyTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        deltaT = [(tt - hourlyTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in hourlyTime]

        def closest_node(node, nodes):
            closest_index = distance.cdist([node], nodes).argmin()
            return nodes[closest_index], closest_index

        simulationsHs = list()
        simulationsTp = list()
        simulationsDm = list()
        simulationsSs = list()
        simulationsTime = list()
        # plus = 900

        for simNum in range(simNum):
            simHs = []
            simTp = []
            simDm = []
            simSs = []
            simTime = []
            simWLTime = []
            print('filling in simulation #{}'.format(simNum))

            for i in range(len(self.simHistBmuChopped[simNum])):
                if np.remainder(i, 1000) == 0:
                    print('done with {} hydrographs'.format(i))
                tempBmu = int(self.simHistBmuChopped[simNum][i] - 1)
                randStorm = random.randint(0, 80000)
                stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                if stormDetails[0] > 8:
                    print('oh boy, we''ve picked a {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                    tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                    randStorm = random.randint(0, 80000)
                    stormDetails = self.gevCopulaSims[tempBmu][randStorm]


                #
                # # IS IT A STORM OR NOT?
                # chanceOfStorm = percentWindows[tempBmu]
                # randChance = random.randint(0, 99999) / 100000
                #
                # if randChance < chanceOfStorm:
                #     print('we have a storm in BMU {}'.format(tempBmu))
                #     if tempBmu == 28:
                #         print('which our copulas say is impossible')
                #         randStorm = random.randint(0, 9999)
                #         stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #     else:
                #         index = np.where((self.gevCopulaSims[tempBmu][:, 0] > 2.5))
                #         subsetGEV = self.gevCopulaSims[tempBmu][index[0], :]
                #         randStorm = random.randint(0, len(index))
                #         stormDetails = subsetGEV[randStorm]
                #     # while stormDetails[0]<2.5:
                #     #     randStorm = random.randint(0, 9999)
                #     #     stormDetails = gevCopulaSims[tempBmu][randStorm]
                #
                # else:
                #     randStorm = random.randint(0, 9999)
                #     stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #     while stormDetails[0] > 2.5:
                #         randStorm = random.randint(0, 9999)
                #         stormDetails = self.gevCopulaSims[tempBmu][randStorm]

                durSim = self.simFutureBmuLengthChopped[simNum][i]

                simDmNorm = (stormDetails[4] - np.asarray(self.bmuDataMin)[tempBmu, 0]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 0] - np.asarray(self.bmuDataMin)[tempBmu, 0])
                simSsNorm = (stormDetails[5] - np.asarray(self.bmuDataMin)[tempBmu, 1]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 1] - np.asarray(self.bmuDataMin)[tempBmu, 1])
                test, closeIndex = closest_node([simDmNorm, simSsNorm], np.asarray(self.bmuDataNormalized)[tempBmu])
                actualIndex = int(np.asarray(self.copulaData[tempBmu])[closeIndex, 7])

                tempHs = ((self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0] - stormDetails[1]) +
                          stormDetails[1]).filled()
                tempTp = ((self.normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2] - stormDetails[3]) +
                          stormDetails[3]).filled()
                tempDm = ((self.normalizedHydros[tempBmu][actualIndex]['dmNorm']) + stormDetails[4])
                tempSs = ((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])

                tempWLtime = np.arange(0,durSim,durSim/len(tempSs))

                if len(self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(
                        self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu, actualIndex))
                if stormDetails[1] < 0:
                    print('woah, we''re less than 0 over here')
                    asdfg
                # if len(tempSs) < len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                #     # print('Ss is shorter than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                #     tempLength = len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'])
                #     tempSs = np.zeros((len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']),))
                #     tempSs[0:len((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])] = (
                #                 (self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])

                # if len(tempSs) > len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    # print('Now Ss is longer than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                    # print('{} vs. {}'.format(len(tempSs),len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])))
                    # tempSs = tempSs[0:-1]
                if len(tempSs) > len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is longer than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempSs = tempSs[0:-1]
                if len(tempSs) < len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is shorter than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempWLtime = tempWLtime[0:-1]


                simHs.append(tempHs)
                simTp.append(tempTp)
                simDm.append(tempDm)
                simSs.append(tempSs)
                # simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                # dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                simTime.append(np.hstack((np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim),
                                          np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim)[-1])))
                simWLTime.append(np.hstack((np.diff(tempWLtime),np.diff(tempWLtime)[-1])))



            cumulativeHours = np.cumsum(np.hstack(simTime))
            # newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
            newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeHours]

            simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyTime]

            # Just for water levels at different time interval
            cumulativeWLHours = np.cumsum(np.hstack(simWLTime))
            newDailyWLTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeWLHours]
            simDeltaWLT = [(tt - newDailyWLTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyWLTime]

            print('water level time vs. surge: {} vs {}'.format(len(np.hstack(simSs)),len(simDeltaWLT)))

            # simData = np.array(
            #     np.vstack((np.hstack(simHs).T, np.hstack(simTp).T, np.hstack(simDm).T, np.hstack(simSs).T)))
            # # simData = np.array((np.ma.asarray(np.hstack(simHs)),np.ma.asarray(np.hstack(simTp)),np.ma.asarray(np.hstack(simDm)),np.ma.asarray(np.hstack(simSs))))
            # # simData = np.array([np.hstack(simHs).filled(),np.hstack(simTp).filled(),np.hstack(simDm).filled(),np.hstack(simSs)])
            #
            # ogdf = pandas.DataFrame(data=simData.T, index=newDailyTime, columns=["hs", "tp", "dm", "ss"])

            print('interpolating')
            interpHs = np.interp(deltaT, simDeltaT, np.hstack(simHs))
            interpTp = np.interp(deltaT, simDeltaT, np.hstack(simTp))
            interpDm = np.interp(deltaT, simDeltaT, np.hstack(simDm))
            interpSs = np.interp(deltaT, simDeltaWLT, np.hstack(simSs))

            badWaves = np.where(interpHs > 10)
            interpHs[badWaves] = interpHs[badWaves]*0+1.5

            simDataInterp = np.array([interpHs, interpTp, interpDm, interpSs])#, interpWind, interpWindDir])

            df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "ss"])
            # df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "ss", "w", "wd"])

            # resampled = df.resample('H')
            # interped = resampled.interpolate()
            # simulationData = interped.values
            # testTime = interped.index  # to_pydatetime()
            # testTime2 = testTime.to_pydatetime()

            # simsPickle = ('/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle'.format(simNum))
            # simsPickle = ('/media/dylananderson/Elements/Sims/simulation{}.pickle'.format(simNum))
            simsPickle = ('historicalSims{}.pickle'.format(simNum))

            outputSims = {}
            outputSims['simulationData'] = simDataInterp.T
            outputSims['df'] = df
            outputSims['simHs'] = np.hstack(simHs)
            outputSims['simTp'] = np.hstack(simTp)
            outputSims['simDm'] = np.hstack(simDm)
            outputSims['simSs'] = np.hstack(simSs)
            outputSims['time'] = hourlyTime

            with open(os.path.join(nodePath,simsPickle), 'wb') as f:
                pickle.dump(outputSims, f)


    def simsHistoricalInterpolatedWinds(self,simNum,nodePath):
        import numpy as np
        from datetime import datetime, date, timedelta
        import random
        from scipy.spatial import distance
        import pickle
        import calendar
        import pandas
        dt = datetime(1979, 6, 1, 0, 0, 0)
        end = datetime(2024, 5, 31, 23, 0, 0)
        step = timedelta(hours=1)
        hourlyTime = []
        while dt < end:
            hourlyTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        deltaT = [(tt - hourlyTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in hourlyTime]

        def closest_node(node, nodes):
            closest_index = distance.cdist([node], nodes).argmin()
            return nodes[closest_index], closest_index

        simulationsHs = list()
        simulationsTp = list()
        simulationsDm = list()
        simulationsSs = list()
        simulationsTime = list()
        simulationsWind = list()
        simulationsWindDir = list()
        # plus = 900

        for simNum in range(simNum):
            simHs = []
            simTp = []
            simDm = []
            simSs = []
            simTime = []
            simWLTime = []
            simU10 = []
            simV10 = []
            print('filling in simulation #{}'.format(simNum))

            for i in range(len(self.simHistBmuChopped[simNum])):
                if np.remainder(i, 1000) == 0:
                    print('done with {} hydrographs'.format(i))
                tempBmu = int(self.simHistBmuChopped[simNum][i] - 1)

                # #  IS IT A STORM OR NOT?
                # chanceOfStorm = percentWindows[tempBmu]
                # randChance = random.randint(0, 99999) / 100000
                #
                # if randChance < chanceOfStorm:
                #     print('we have a storm in BMU {}'.format(tempBmu))
                #     if tempBmu == 28:
                #         print('which our copulas say is impossible')
                #         randStorm = random.randint(0, 90000)
                #         stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #     else:
                #         index = np.where((self.gevCopulaSims[tempBmu][:, 0] > 2.5))
                #         subsetGEV = self.gevCopulaSims[tempBmu][index[0], :]
                #         randStorm = random.randint(0, len(index))
                #         stormDetails = subsetGEV[randStorm]
                #     # while stormDetails[0]<2.5:
                #     #     randStorm = random.randint(0, 9999)
                #     #     stormDetails = gevCopulaSims[tempBmu][randStorm]
                #
                # else:
                #     randStorm = random.randint(0, 90000)
                #     stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                #     while stormDetails[0] > 2.75:
                #         randStorm = random.randint(0, 90000)
                #         stormDetails = self.gevCopulaSims[tempBmu][randStorm]


                randStorm = random.randint(0, 80000)
                stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                if stormDetails[0] > 8:
                    print('oh boy, we''ve picked a {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                    tempBmu = int(self.simFutureBmuChopped[simNum][i] - 1)
                    randStorm = random.randint(0, 80000)
                    stormDetails = self.gevCopulaSims[tempBmu][randStorm]
                if stormDetails[0] > 8:
                    print('yikes, we picked another {} m storm wave in BMU #{}'.format(stormDetails[0], tempBmu))
                    index = np.where((self.gevCopulaSims[tempBmu][:, 0] < 2.5))
                    subsetGEV = self.gevCopulaSims[tempBmu][index[0], :]
                    randStorm = random.randint(0, len(index))
                    stormDetails = subsetGEV[randStorm]


                durSim = self.simFutureBmuLengthChopped[simNum][i]/(24/self.avgTime)

                simDmNorm = (stormDetails[8] - np.asarray(self.bmuDataMin)[tempBmu, 0]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 0] - np.asarray(self.bmuDataMin)[tempBmu, 0])
                simSsNorm = (stormDetails[9] - np.asarray(self.bmuDataMin)[tempBmu, 1]) / (
                            np.asarray(self.bmuDataMax)[tempBmu, 1] - np.asarray(self.bmuDataMin)[tempBmu, 1])
                test, closeIndex = closest_node([simDmNorm, simSsNorm], np.asarray(self.bmuDataNormalized)[tempBmu])
                actualIndex = int(np.asarray(self.copulaData[tempBmu])[closeIndex, 11])

                tempHs = ((self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0] - stormDetails[1]) +
                          stormDetails[1]).filled()
                tempTp = ((self.normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2] - stormDetails[3]) +
                          stormDetails[3]).filled()
                tempU = ((self.normalizedHydros[tempBmu][actualIndex]['uNorm']) * (stormDetails[4] - stormDetails[5]) +
                            stormDetails[5])
                tempV = ((self.normalizedHydros[tempBmu][actualIndex]['vNorm']) * (stormDetails[6] - stormDetails[7]) +
                            stormDetails[7])

                tempDm = ((self.normalizedHydros[tempBmu][actualIndex]['dmNorm']) + stormDetails[8])
                tempSs = ((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[9])


                tempWLtime = np.arange(0,durSim,durSim/len(tempSs))

                if len(self.normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(
                        self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu, actualIndex))
                if stormDetails[1] < 0:
                    print('woah, we''re less than 0 over here')
                    asdfg
                # if len(tempSs) < len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                #     # print('Ss is shorter than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                #     tempLength = len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'])
                #     tempSs = np.zeros((len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']),))
                #     tempSs[0:len((self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])] = (
                #                 (self.normalizedHydros[tempBmu][actualIndex]['ntrNorm']) + stormDetails[5])

                # if len(tempSs) > len(self.normalizedHydros[tempBmu][actualIndex]['timeNorm']):
                    # print('Now Ss is longer than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
                    # print('{} vs. {}'.format(len(tempSs),len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])))
                    # tempSs = tempSs[0:-1]
                if len(tempSs) > len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is longer than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempSs = tempSs[0:-1]
                if len(tempSs) < len(tempWLtime):
                    # print('Now Ss is longer than WL Time in bmu {}, index {}'.format(tempBmu, actualIndex))
                    # print('Now Ss is shorter than WL Time: {} vs {}'.format(len(tempSs), len(tempWLtime)))
                    tempWLtime = tempWLtime[0:-1]


                simHs.append(tempHs)
                simTp.append(tempTp)
                simDm.append(tempDm)
                simSs.append(tempSs)
                simU10.append(tempU)
                simV10.append(tempV)
                # simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                # dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
                simTime.append(np.hstack((np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim),
                                          np.diff(self.normalizedHydros[tempBmu][actualIndex]['timeNorm'] * durSim)[-1])))
                if len(tempSs)>1:
                    simWLTime.append(np.hstack((np.diff(tempWLtime),np.diff(tempWLtime)[-1])))
                else:
                    #print(tempSs)
                    simWLTime.append(durSim)



            cumulativeHours = np.cumsum(np.hstack(simTime))
            newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
            # newDailyTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeHours]

            # simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyTime]
            simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600) for tt in newDailyTime]

            # Just for water levels at different time interval
            cumulativeWLHours = np.cumsum(np.hstack(simWLTime))
            newDailyWLTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii) for ii in cumulativeWLHours]
            # newDailyWLTime = [datetime(self.futureSimStart, 6, 1) + timedelta(days=ii*(24/self.avgTime)) for ii in cumulativeWLHours]
            simDeltaWLT = [(tt - newDailyWLTime[0]).total_seconds() / (3600) for tt in newDailyWLTime]
            # simDeltaWLT = [(tt - newDailyWLTime[0]).total_seconds() / (3600 * (24/self.avgTime)) for tt in newDailyWLTime]

            print('water level time vs. surge: {} vs {}'.format(len(np.hstack(simSs)),len(simDeltaWLT)))

            # simData = np.array(
            #     np.vstack((np.hstack(simHs).T, np.hstack(simTp).T, np.hstack(simDm).T, np.hstack(simSs).T)))
            # # simData = np.array((np.ma.asarray(np.hstack(simHs)),np.ma.asarray(np.hstack(simTp)),np.ma.asarray(np.hstack(simDm)),np.ma.asarray(np.hstack(simSs))))
            # # simData = np.array([np.hstack(simHs).filled(),np.hstack(simTp).filled(),np.hstack(simDm).filled(),np.hstack(simSs)])
            #
            # ogdf = pandas.DataFrame(data=simData.T, index=newDailyTime, columns=["hs", "tp", "dm", "ss"])

            print('interpolating')
            interpHs = np.interp(deltaT, simDeltaT, np.hstack(simHs))
            interpTp = np.interp(deltaT, simDeltaT, np.hstack(simTp))
            interpDm = np.interp(deltaT, simDeltaT, np.hstack(simDm))
            interpU10 = np.interp(deltaT, simDeltaT, np.hstack(simU10))
            interpV10 = np.interp(deltaT, simDeltaT, np.hstack(simV10))
            interpSs = np.interp(deltaT, simDeltaWLT, np.hstack(simSs))

            # badWaves = np.where(interpHs > 10)
            # interpHs[badWaves] = interpHs[badWaves]*0+1.5

            simDataInterp = np.array([interpHs, interpTp, interpDm, interpU10, interpV10, interpSs])#, interpWind, interpWindDir])

            df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm","u10","v10", "ss"])
            # df = pandas.DataFrame(data=simDataInterp.T, index=hourlyTime, columns=["hs", "tp", "dm", "ss", "w", "wd"])

            # resampled = df.resample('H')
            # interped = resampled.interpolate()
            # simulationData = interped.values
            # testTime = interped.index  # to_pydatetime()
            # testTime2 = testTime.to_pydatetime()

            # simsPickle = ('/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle'.format(simNum))
            # simsPickle = ('/media/dylananderson/Elements/Sims/simulation{}.pickle'.format(simNum))
            simsPickle = ('historicalSims{}.pickle'.format(simNum))

            outputSims = {}
            outputSims['simulationData'] = simDataInterp.T
            outputSims['df'] = df
            outputSims['simHs'] = np.hstack(simHs)
            outputSims['simTp'] = np.hstack(simTp)
            outputSims['simDm'] = np.hstack(simDm)
            outputSims['simSs'] = np.hstack(simSs)
            outputSims['simU10'] = np.hstack(simU10)
            outputSims['simV10'] = np.hstack(simV10)

            outputSims['time'] = hourlyTime

            with open(os.path.join(nodePath,simsPickle), 'wb') as f:
                pickle.dump(outputSims, f)


    def simsFutureValidated(self,met,numSims=5,threshold=8,hsthreshold=3):
        import numpy as np
        from datetime import datetime, date, timedelta
        import random
        import pandas as pd
        from functions import return_value
        import matplotlib.pyplot as plt
        from dateutil.relativedelta import relativedelta

        tC = met.timeWave
        data = np.array([met.Hs, met.Tp, met.Dm, met.u10, met.v10])
        ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm", "u10","v10"])
        year = np.array([tt.year for tt in tC])
        ogdf['year'] = year
        month = np.array([tt.month for tt in tC])
        ogdf['month'] = month

        dailyMaxHs = ogdf.resample("d")['hs'].max()
        seasonalMean = ogdf.groupby('month').mean()
        seasonalStd = ogdf.groupby('month').std()
        yearlyMax = ogdf.groupby('year').max()

        c = 0
        fourDayMax = []
        while c < len(met.Hs):
            fourDayMax.append(np.nanmax(met.Hs[c:c + 96]))
            c = c + 96
        fourDayMaxHs = np.asarray(fourDayMax)

        simSeasonalMean = np.nan * np.ones((numSims, 12))
        simSeasonalStd = np.nan * np.ones((numSims, 12))
        simYearlyMax = np.nan * np.ones((numSims, 101))

        yearArray = []
        zNArray = []
        ciArray = []
        for hh in range(numSims):

            file = r"futureSims{}.pickle".format(hh)

            with open(os.path.join(self.savePath,file), "rb") as input_file:
                # simsInput = pickle.load(input_file)
                simsInput = pd.read_pickle(input_file)
            simulationData = simsInput['simulationData']
            # df = simsInput['df']
            time = simsInput['time']

            badInd2 = np.where(simulationData[:, 0] > threshold)
            simulationData[badInd2, 0] = np.nan
            simulationData[badInd2, 1] = np.nan
            simulationData[badInd2, 2] = np.nan
            simulationData[badInd2, 3] = np.nan
            simulationData[badInd2, 4] = np.nan
            dfdata = simulationData
            df = pd.DataFrame(data=dfdata, index=time, columns=["hs", "tp", "dm","u10","v10","ss"])
            year = np.array([tt.year for tt in time])
            df['year'] = year
            month = np.array([tt.month for tt in time])
            df['month'] = month

            g1 = df.groupby(pd.Grouper(freq="M")).mean()
            simSeasonalMean[hh, :] = df.groupby('month').mean()["hs"]
            simSeasonalStd[hh, :] = df.groupby('month').std()["hs"]
            simYearlyMax[hh, :] = df.groupby('year').max()["hs"]
            dailyMaxHsSim = df.resample("d")["hs"].max()

            fourtyTwoYears = 42*365.25*24
            c = 0
            fourDayMaxSim = []
            while c < len(simulationData):
                fourDayMaxSim.append(np.nanmax(simulationData[c:c + 96, 0]))
                c = c + 96
            fourDayMaxHsSim = np.asarray(fourDayMaxSim)
            # print(len(np.asarray(fourDayMaxHsSim)))

            # sim = return_value(np.asarray(fourDayMaxHsSim)[0:365*41], 15, 0.05, 365/4, 36525/4, 'mle')
            sim = return_value(np.asarray(fourDayMaxHsSim)[0:int(365/4*41)], hsthreshold, 0.05, 365 / 4, 36525 / 4, 'mle')

            yearArray.append(sim['year_array'])
            zNArray.append(sim['z_N'])
            ciArray.append(sim['CI'])

        dt = datetime(2002, 1, 1)
        end = datetime(2003, 1, 1)
        # dt = datetime(2022, 1, 1)
        # end = datetime(2023, 1, 1)
        step = relativedelta(months=1)
        plotTime = []
        while dt < end:
            plotTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        # print(len(np.asarray(fourDayMax)))
        historical = return_value(np.asarray(fourDayMax), hsthreshold, 0.05, 365 / 4, 36525 / 4, 'mle')
        #historical2 = return_value(np.asarray(fourDayMaxHsSim), hsthreshold, 0.05, 365 / 4, 36525 / 4, 'mle')


        var = 'hs'
        plt.figure()
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        ax1.plot(plotTime, seasonalMean[var], label='WIS record (42 years)')
        ax1.fill_between(plotTime, seasonalMean[var] - seasonalStd[var], seasonalMean[var] + seasonalStd[var],
                         color='b', alpha=0.2)
        ax1.plot(plotTime, df.groupby('month').mean()[var], label='Synthetic record (100 years)')
        ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').std()[var],
                         df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').percentile()[var],
        #                  df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
        ax1.set_xticks(
            [plotTime[0], plotTime[1], plotTime[2], plotTime[3], plotTime[4], plotTime[5], plotTime[6], plotTime[7],
             plotTime[8], plotTime[9], plotTime[10], plotTime[11]])
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.legend()
        ax1.set_ylabel('hs (m)')
        ax1.set_title('Seasonal wave variability')


        var = 'v10'
        plt.figure()
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        ax1.plot(plotTime, seasonalMean[var], label='WIS record (42 years)')
        ax1.fill_between(plotTime, seasonalMean[var] - seasonalStd[var], seasonalMean[var] + seasonalStd[var],
                         color='b', alpha=0.2)
        ax1.plot(plotTime, df.groupby('month').mean()[var], label='Synthetic record (100 years)')
        ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').std()[var],
                         df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').percentile()[var],
        #                  df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
        ax1.set_xticks(
            [plotTime[0], plotTime[1], plotTime[2], plotTime[3], plotTime[4], plotTime[5], plotTime[6], plotTime[7],
             plotTime[8], plotTime[9], plotTime[10], plotTime[11]])
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.legend()
        ax1.set_ylabel('v10 (m/s)')
        ax1.set_title('Seasonal wave variability')

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # plt.style.use('dark_background')

        # to do order this by uncertainty
        plt.figure(8)
        colorparam = np.zeros((len(zNArray),))
        for qq in range(len(zNArray)):
            normalize = mcolors.Normalize(vmin=0, vmax=5)
            colorparam[qq] = ciArray[qq]
            colormap = cm.Greys_r
            color = colormap(normalize(colorparam[qq]))
            plt.plot(yearArray[qq], zNArray[qq], color=color, alpha=0.75)  # color=[0.5,0.5,0.5],alpha=0.5)

        plt.plot(historical['year_array'], historical['CI_z_N_high_year'], linestyle='--', color='red', alpha=0.8,
                 lw=0.9, label='Confidence Bands')
        plt.plot(historical['year_array'], historical['CI_z_N_low_year'], linestyle='--', color='red', alpha=0.8,
                 lw=0.9)
        plt.plot(historical['year_array'], historical['z_N'], color='orange', label='Theoretical Return Level')
        plt.scatter(historical['N'], historical['sample_over_thresh'], color='orange', label='Empirical Return Level',
                    zorder=10)
        # plt.plot(historical2['year_array'], historical2['z_N'], color='black', label='Theoretical Return Level')
        # plt.scatter(historical2['N'], historical2['sample_over_thresh'][0:-1], color='orange', label='Empirical Return Level',
        #             zorder=10)
        plt.xscale('log')
        plt.xlabel('Return Period (years)')
        plt.ylabel('Return Level (m)')
        plt.title('Return Level Plot (Wave Height)')
        plt.legend()

        plt.show()



    def simsHistoricalValidated(self,met,numSims=5,threshold=8,hsthreshold=3):
        import numpy as np
        from datetime import datetime, date, timedelta
        import random
        import pandas as pd
        from functions import return_value
        import matplotlib.pyplot as plt
        from dateutil.relativedelta import relativedelta

        tC = met.timeWave
        data = np.array([met.Hs, met.Tp, met.Dm, met.u10, met.v10])
        ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm", "u10","v10"])
        year = np.array([tt.year for tt in tC])
        ogdf['year'] = year
        month = np.array([tt.month for tt in tC])
        ogdf['month'] = month

        dailyMaxHs = ogdf.resample("d")['hs'].max()
        seasonalMean = ogdf.groupby('month').mean()
        seasonalStd = ogdf.groupby('month').std()
        yearlyMax = ogdf.groupby('year').max()

        c = 0
        fourDayMax = []
        while c < len(met.Hs):
            fourDayMax.append(np.nanmax(met.Hs[c:c + 96]))
            c = c + 96
        fourDayMaxHs = np.asarray(fourDayMax)

        simSeasonalMean = np.nan * np.ones((numSims, 12))
        simSeasonalStd = np.nan * np.ones((numSims, 12))
        simYearlyMax = np.nan * np.ones((numSims, 46))

        yearArray = []
        zNArray = []
        ciArray = []
        for hh in range(numSims):

            file = r"historicalSims{}.pickle".format(hh)

            with open(os.path.join(self.savePath,file), "rb") as input_file:
                # simsInput = pickle.load(input_file)
                simsInput = pd.read_pickle(input_file)
            simulationData = simsInput['simulationData']
            # df = simsInput['df']
            time = simsInput['time']

            badInd2 = np.where(simulationData[:, 0] > threshold)
            simulationData[badInd2, 0] = np.nan
            simulationData[badInd2, 1] = np.nan
            simulationData[badInd2, 2] = np.nan
            simulationData[badInd2, 3] = np.nan
            simulationData[badInd2, 4] = np.nan
            dfdata = simulationData
            df = pd.DataFrame(data=dfdata, index=time, columns=["hs", "tp", "dm","u10","v10","ss"])
            year = np.array([tt.year for tt in time])
            df['year'] = year
            month = np.array([tt.month for tt in time])
            df['month'] = month

            g1 = df.groupby(pd.Grouper(freq="M")).mean()
            simSeasonalMean[hh, :] = df.groupby('month').mean()["hs"]
            simSeasonalStd[hh, :] = df.groupby('month').std()["hs"]
            simYearlyMax[hh, :] = df.groupby('year').max()["hs"]
            dailyMaxHsSim = df.resample("d")["hs"].max()

            #fourtyTwoYears = 42*365.25*24
            c = 0
            fourDayMaxSim = []
            while c < len(simulationData):
                fourDayMaxSim.append(np.nanmax(simulationData[c:c + 96, 0]))
                c = c + 96
            fourDayMaxHsSim = np.asarray(fourDayMaxSim)
            # print(len(np.asarray(fourDayMaxHsSim)))

            # sim = return_value(np.asarray(fourDayMaxHsSim)[0:365*41], 15, 0.05, 365/4, 36525/4, 'mle')
            sim = return_value(np.asarray(fourDayMaxHsSim)[0:int(365/4*45)], hsthreshold, 0.05, 365 / 4, (365*45) / 4, 'mle')

            yearArray.append(sim['year_array'])
            zNArray.append(sim['z_N'])
            ciArray.append(sim['CI'])

        dt = datetime(2002, 1, 1)
        end = datetime(2003, 1, 1)
        # dt = datetime(2022, 1, 1)
        # end = datetime(2023, 1, 1)
        step = relativedelta(months=1)
        plotTime = []
        while dt < end:
            plotTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        # print(len(np.asarray(fourDayMax)))
        # historical = return_value(np.asarray(fourDayMax), hsthreshold, 0.05, 365 / 4, 36525 / 4, 'mle')
        #historical2 = return_value(np.asarray(fourDayMaxHsSim), hsthreshold, 0.05, 365 / 4, 36525 / 4, 'mle')
        historical = return_value(np.asarray(fourDayMax), hsthreshold, 0.05, 365 / 4, (365*45) / 4, 'mle')


        var = 'hs'
        plt.figure()
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        ax1.plot(plotTime, seasonalMean[var], label='WIS record (42 years)')
        ax1.fill_between(plotTime, seasonalMean[var] - seasonalStd[var], seasonalMean[var] + seasonalStd[var],
                         color='b', alpha=0.2)
        ax1.plot(plotTime, df.groupby('month').mean()[var], label='Synthetic record (100 years)')
        ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').std()[var],
                         df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').percentile()[var],
        #                  df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
        ax1.set_xticks(
            [plotTime[0], plotTime[1], plotTime[2], plotTime[3], plotTime[4], plotTime[5], plotTime[6], plotTime[7],
             plotTime[8], plotTime[9], plotTime[10], plotTime[11]])
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.legend()
        ax1.set_ylabel('hs (m)')
        ax1.set_title('Seasonal wave variability')


        var = 'v10'
        plt.figure()
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        ax1.plot(plotTime, seasonalMean[var], label='WIS record (42 years)')
        ax1.fill_between(plotTime, seasonalMean[var] - seasonalStd[var], seasonalMean[var] + seasonalStd[var],
                         color='b', alpha=0.2)
        ax1.plot(plotTime, df.groupby('month').mean()[var], label='Synthetic record (100 years)')
        ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').std()[var],
                         df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, df.groupby('month').mean()[var] - df.groupby('month').percentile()[var],
        #                  df.groupby('month').mean()[var] + df.groupby('month').std()[var], color='orange', alpha=0.2)
        # ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
        ax1.set_xticks(
            [plotTime[0], plotTime[1], plotTime[2], plotTime[3], plotTime[4], plotTime[5], plotTime[6], plotTime[7],
             plotTime[8], plotTime[9], plotTime[10], plotTime[11]])
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.legend()
        ax1.set_ylabel('v10 (m/s)')
        ax1.set_title('Seasonal wave variability')

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # plt.style.use('dark_background')

        # to do order this by uncertainty
        plt.figure(8)
        colorparam = np.zeros((len(zNArray),))
        for qq in range(len(zNArray)):
            normalize = mcolors.Normalize(vmin=0, vmax=5)
            colorparam[qq] = ciArray[qq]
            colormap = cm.Greys_r
            color = colormap(normalize(colorparam[qq]))
            plt.plot(yearArray[qq], zNArray[qq], color=color, alpha=0.75)  # color=[0.5,0.5,0.5],alpha=0.5)

        plt.plot(historical['year_array'], historical['CI_z_N_high_year'], linestyle='--', color='red', alpha=0.8,
                 lw=0.9, label='Confidence Bands')
        plt.plot(historical['year_array'], historical['CI_z_N_low_year'], linestyle='--', color='red', alpha=0.8,
                 lw=0.9)
        plt.plot(historical['year_array'], historical['z_N'], color='orange', label='Theoretical Return Level')
        plt.scatter(historical['N'], historical['sample_over_thresh'], color='orange', label='Empirical Return Level',
                    zorder=10)
        # plt.plot(historical2['year_array'], historical2['z_N'], color='black', label='Theoretical Return Level')
        # plt.scatter(historical2['N'], historical2['sample_over_thresh'][0:-1], color='orange', label='Empirical Return Level',
        #             zorder=10)
        plt.xscale('log')
        plt.xlabel('Return Period (years)')
        plt.ylabel('Return Level (m)')
        plt.title('Return Level Plot (Wave Height)')
        plt.legend()

        plt.show()