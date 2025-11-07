
import numpy as np
import os
from functions import loadWIS, loadWaterLevel
from datetime import datetime, timedelta

class getMetOcean():
    '''
    Class to get metOcean Data
    '''

    def __init__(self, **kwargs):
        self.lonLeft = kwargs.get('lonLeft', 275)
        self.lonRight = kwargs.get('lonRight',350)
        self.latBot = kwargs.get('latBot', 15)
        self.latTop = kwargs.get('latTop', 50)
        self.avgTime = kwargs.get('avgTime',24)
        self.startTime = kwargs.get('startTime',[1979,1,1])
        self.endTime = kwargs.get('endTime',[2020,12,31])
        self.wisPath = kwargs.get('wisPath','/users/dylananderson/documents/data/WIS_ST63218/')
        self.wlPath = kwargs.get('wlPath','/users/dylananderson/documents/data/frfWaterLevel/')
        self.shoreNormal = kwargs.get('shoreNormal',72)
        self.chlDataLoc = kwargs.get('chlDataLoc',u'https://chldata.erdc.dren.mil/thredds/dodsC/wis/')
        self.savePath = kwargs.get('savePath',os.getcwd())


    def getWaterLevels(self):
        # GETTING WATER LEVELS
        wlfiles = os.listdir(self.wlPath)
        wlfiles.sort()
        wlfiles_path = [os.path.join(os.path.abspath(self.wlPath), x) for x in wlfiles]


        fileDates = []
        for hh in range(len(wlfiles)):
            fileYear = int(wlfiles[hh].split('_')[-1].split('.')[0][0:4])
            fileMonth = int(wlfiles[hh].split('_')[-1].split('.')[0][4:6])
            fileDates.append(datetime(fileYear,fileMonth,1))

        ind = np.where((np.asarray(fileDates) >= datetime(self.startTime[0],self.startTime[1],1)) & (np.asarray(fileDates) <= datetime(self.endTime[0],self.endTime[1],1)))

        wlfiles = np.asarray(wlfiles)[ind]
        wlfiles_path = np.asarray(wlfiles_path)[ind]

        timeWaterLevelFRF = []
        waterLevelFRF = []
        predictedWaterLevelFRF = []
        residualWaterLevelFRF = []
        for i in wlfiles_path:
            waterLevels = loadWaterLevel(i)
            waterLevelFRF = np.append(waterLevelFRF, waterLevels['waterLevel'])
            predictedWaterLevelFRF = np.append(predictedWaterLevelFRF, waterLevels['predictedWaterLevel'])
            residualWaterLevelFRF = np.append(residualWaterLevelFRF, waterLevels['residualWaterLevel'])
            timeWaterLevelFRF = np.append(timeWaterLevelFRF, waterLevels['time'].flatten())
        tWaterLevelFRF = np.asarray([datetime.fromtimestamp(x) for x in timeWaterLevelFRF])
        # Need to remove times when NOAA has flagged a bad water level
        badWaterLevel = np.where((np.asarray(residualWaterLevelFRF) < -99))
        wl = np.asarray(waterLevelFRF)
        wl[badWaterLevel] = wl[badWaterLevel] * np.nan
        predWl = np.asarray(predictedWaterLevelFRF)
        predWl[badWaterLevel] = predWl[badWaterLevel] * np.nan
        resWl = np.asarray(residualWaterLevelFRF)
        resWl[badWaterLevel] = resWl[badWaterLevel]*np.nan

        nanInd = np.isnan(wl)
        wl = wl[~nanInd]
        predWl = predWl[~nanInd]
        resWl = resWl[~nanInd]
        tWaterLevelFRF = tWaterLevelFRF[~nanInd]

        self.waterLevel = wl
        self.predWL = predWl
        self.resWL = resWl
        self.timeWL = tWaterLevelFRF


    def getDailySurge(self,latSS,lonSS,wts, loadPrior = False,loadPickle = './'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                cops = pickle.load(input_file)
            self.resWl = cops['resWl']
            self.timeWL = cops['timeWL']
            print('loaded prior water levels')

        else:
            import numpy as np
            print('Getting closest GTSM node')

            # latSS = 39.1667
            # lonSS = -74.3333

            import netCDF4
            nc = netCDF4.Dataset('/Users/dylananderson/Downloads/surge_daily_max.nc')
            xCoords = nc.variables['station_x_coordinate'][:]
            yCoords = nc.variables['station_y_coordinate'][:]

            import datetime as dt
            st = dt.datetime(1979, 1, 1)
            end = dt.datetime(2015, 1, 1)
            from dateutil.relativedelta import relativedelta
            step = relativedelta(days=1)
            dailyTimeEval = []
            while st < end:
                dailyTimeEval.append(st)  # .strftime('%Y-%m-%d'))
                st += step



            from math import cos, asin, sqrt

            def distance(lat1, lon1, lat2, lon2):
                p = 0.017453292519943295
                hav = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
                return 12742 * asin(sqrt(hav))

            tempDataList = [{'lat':yCoords[pp], 'lon':xCoords[pp]} for pp in range(len(xCoords))]
            def closest(data, v):
                return min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))

            v = {'lat': latSS, 'lon': lonSS}
            out = closest(tempDataList, v)
            index = np.where((xCoords==out['lon']) & (yCoords==out['lat']))
            if len(index[0]) > 1:
                ss = nc.variables['storm_surge'][:, index[0][0]][:].flatten()
            else:
                ss = nc.variables['storm_surge'][:,index[0]][:].flatten()

            print('extrapolating with SLP PCs')
            dailyTime = dailyTimeEval[31:]
            dailySS = ss[31:]
            # plt.figure()
            # plt.plot(xCoords,yCoords,'.')
            # plt.plot(xCoords[index[0]],yCoords[index[0]],'o')
            # plt.figure()
            # plt.plot(dailyTimeEval,ss)

            import pandas as pd
            df = pd.DataFrame(data=wts.PCs[:,0:200],index=wts.DATES)
            dailySLPs = df.resample("d").mean()
            dailyTrimmedSLPs = dailySLPs.loc[dailyTime[0]:dailyTime[-1],:]

            nanSS = np.where(np.isnan(dailySS))
            dailyVars = np.delete(dailyTrimmedSLPs.values, nanSS[0], axis=0)
            ssVars = np.delete(dailySS, nanSS[0])

            predVars = wts.PCs[:,0:200]
            predTime = wts.DATES

            from sklearn.linear_model import LinearRegression
            allPCsToTry = np.arange(0, 199)
            bestPC = []
            improvingScoreSS = []
            for qq in range(50):
                allScoresSS = []
                for yy in range(len(allPCsToTry)):
                    if qq == 0:
                        tryPCs = yy
                        xSS = dailyVars[:, yy].reshape((-1, 1))
                        ySS = np.array(ssVars)


                    else:
                        tryPCs = np.hstack([np.asarray(bestPC).flatten(), np.asarray(allPCsToTry[yy]).flatten()])

                        xSS = dailyVars[:, tryPCs] #dailyTrimmedSLPs[:, tryPCs]  # .reshape((-1,1))
                        ySS = np.array(ssVars)

                    modelSS = LinearRegression().fit(xSS, ySS)

                    r_sqSS = modelSS.score(xSS, ySS)

                    # print(f"coefficient of determination: {r_sq}")
                    # print(f"intercept: {model.intercept_}")

                    allScoresSS.append(r_sqSS)

                bestAdditionIndex = np.argmax(np.array(allScoresSS))
                bestAddition = allPCsToTry[bestAdditionIndex]
                bestPC.append(bestAddition)
                removeIndex = np.where(allPCsToTry == bestAddition)
                allPCsToTry = np.delete(allPCsToTry, removeIndex)
                improvingScoreSS.append(np.max(np.array(allScoresSS)))
                print('Iter {}: Adding PC#{}, cumulative score:{}'.format(qq, bestAddition,
                                                                          np.max(np.array(allScoresSS))))
            import matplotlib.pyplot as plt
            # plt.figure()
            # ax1 = plt.subplot2grid((2, 3), (0, 1))
            # ax1.plot(ySS, modelSS.predict(xSS), '.')
            # ax1.plot([-1.5, 2], [-1.5, 2], 'k--')
            # ax1.plot([-1.5, 2], [0, 0], 'k--')
            # ax1.plot([0, 0], [-1.5, 2], 'k--')
            # ax1.set_xlabel('GTSM (m, MSL)')
            # ax1.set_ylabel('Predicted Non-tidal Residual (m, MSL)')
            #
            # ax1b = plt.subplot2grid((2, 3), (1, 1))
            ySSStd = np.std(ySS)
            ySSMean = np.mean(ySS)
            yModSS = modelSS.predict(xSS)
            yModSSStd = np.std(yModSS)
            yModSSMean = np.mean(yModSS)
            #
            # ax1b.plot(ySS, ((yModRed - yModRedMean) / yModRedStd) * yRedStd + yRedMean, '.')
            # ax1b.plot([-1.5, 2], [-1.5, 2], 'k--')
            # ax1b.plot([-1.5, 2], [0, 0], 'k--')
            # ax1b.plot([0, 0], [-1.5, 2], 'k--')
            # ax1b.set_xlabel('GTSM (m, MSL)')
            # ax1b.set_ylabel('Predicted Non-tidal Residual (m, MSL)')
            # plt.figure()
            # ax10 = plt.subplot2grid((1, 1), (0, 0))
            # # plt.plot(timeData,ss+mmsla+seasonal)
            # ax10.plot(predTime,
            #           ((modelSS.predict(predVars[:, bestPC]) - yModSSMean) / yModSSStd) * ySSStd + ySSMean)
            # ax10.set_xlabel('time')
            # ax10.set_ylabel('NTR Prediction')
            resWl = ((modelSS.predict(predVars[:, bestPC]) - yModSSMean) / yModSSStd) * ySSStd + ySSMean

            self.resWl = resWl
            self.timeWL = predTime
            import pickle
            samplesPickle = 'surges.pickle'
            outputSamples = {}
            outputSamples['resWl'] = self.resWl
            outputSamples['timeWL'] = self.timeWL
            with open(os.path.join(self.savePath, samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)



    def getWISLocal(self, loadPrior = False,loadPickle = './'):

        if loadPrior == True:
            import numpy as np
            import pickle
            with open(loadPickle, "rb") as input_file:
                cops = pickle.load(input_file)

            self.timeWave = cops['timeWave']
            self.Hs = cops['Hs']
            self.Tp = cops['Tp']
            self.Dm = cops['Dm']
            # self.timeWind = cops['timeWind']
            self.u10 = cops['u10']
            self.v10 = cops['v10']

            print('loaded prior waves and winds')

        else:
            import numpy as np
            import os
            # LOADING THE WIS FILES
            # Need to sort the files to ensure correct temporal order...
            # waveFiles = os.listdir(self.wisPath)
            waveFiles = [filename for filename in os.listdir(self.wisPath) if filename.startswith("WIS")]

            waveFiles.sort()
            waveFiles_path = [os.path.join(os.path.abspath(self.wisPath), x) for x in waveFiles]


            fileDates = []
            for hh in range(len(waveFiles)):
                fileYear = int(waveFiles[hh].split('_')[-1].split('.')[0][0:4])
                fileMonth = int(waveFiles[hh].split('_')[-1].split('.')[0][4:6])
                fileDates.append(datetime(fileYear,fileMonth,1))

            ind = np.where((np.asarray(fileDates) >= datetime(self.startTime[0],self.startTime[1],1)) & (np.asarray(fileDates) <= datetime(self.endTime[0],self.endTime[1],1)))

            waveFiles = np.asarray(waveFiles)[ind]
            waveFiles_path =np.asarray( waveFiles_path)[ind]

            Hs = []
            Tp = []
            Dm = []
            wS = []
            wD = []
            sp = []

            timeWave = []
            for i in waveFiles_path:
                waves = loadWIS(i)
                Hs = np.append(Hs, waves['waveHs'])
                Tp = np.append(Tp, waves['waveTp'])
                Dm = np.append(Dm, waves['waveMeanDirection'])
                wS = np.append(wS, waves['windSpeed'])
                wD = np.append(wD, waves['windDirection'])
                sp = np.append(sp, waves['waveSpread'])
                timeWave = np.append(timeWave, waves['t'].flatten())

            # tWave = [DT.datetime.fromtimestamp(x) for x in timeWave]
            tWave = [datetime.fromtimestamp(x) for x in timeWave]
            tC = np.array(tWave)
            # reorienting wave directions to FRF's shore normal (72 degrees)
            waveNorm = np.asarray(Dm) - int(self.shoreNormal)
            neg = np.where((waveNorm > 180))
            waveNorm[neg[0]] = waveNorm[neg[0]] - 360
            pos = np.where((waveNorm < -180))
            waveNorm[pos[0]] = waveNorm[pos[0]] + 360
            # offpos = np.where((waveNorm > 90))
            # offneg = np.where((waveNorm < -90))
            # waveNorm[offpos[0]] = waveNorm[offpos[0]] * 0
            # waveNorm[offneg[0]] = waveNorm[offneg[0]] * 0

            # reorienting wave directions to u and v in polar based on shoreline orientation
            windNorm = np.asarray(wD) - int(self.shoreNormal) # making zero the shorenormal
            negW = np.where((windNorm > 180))
            windNorm[negW[0]] = windNorm[negW[0]] - 360
            posW = np.where((windNorm < -180))
            windNorm[posW[0]] = windNorm[posW[0]] + 360
            windNorm = -windNorm # flipping from compass to polar
            u10 = np.cos(windNorm*np.pi/180)*wS
            v10 = np.sin(windNorm*np.pi/180)*wS

            self.timeWave = tC
            self.Hs = Hs
            self.Tp = Tp
            self.Dm = waveNorm
            self.u10 = u10
            self.v10 = v10
            self.sp = sp

            import pickle
            samplesPickle = 'metOcean.pickle'
            outputSamples = {}
            outputSamples['timeWave'] = self.timeWave
            outputSamples['Hs'] = self.Hs
            outputSamples['Tp'] = self.Tp
            outputSamples['Dm'] = self.Dm
            # outputSamples['timeWind'] = self.timeWind
            outputSamples['u10'] = self.u10
            outputSamples['v10'] = self.v10
            outputSamples['sp'] = self.sp

            with open(os.path.join(self.savePath, samplesPickle), 'wb') as f:
                pickle.dump(outputSamples, f)

    def getWISThredds(self,basin,buoy,**kwargs):
        from posixpath import join as urljoin
        import xarray as xr
        import time
        from datetime import timedelta

        if 'variable' in kwargs:


            variable = kwargs['variable']

            t1 = time.time()



            if self.startTime[0] == self.endTime[0]:
                months = np.arange(self.startTime[1], self.endTime[1] + 1)
                counter = 0
                for hh in months:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.startTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.startTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.startTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)

                    if counter == 0:
                        df = ds[variable]
                    else:
                        df = xr.concat([df, ds[variable]], dim='time')
                        counter = counter + 1

            elif self.startTime[0] == (self.endTime[0] - 1):
                months1 = np.arange(self.startTime[1], 13)
                counter = 0
                for hh in months1:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.startTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.startTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.startTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)

                    if counter == 0:
                        df = ds[variable]
                    else:
                        df = xr.concat([df, ds[variable]], dim='time')
                        counter = counter + 1
                months2 = np.arange(1, self.endTime[1] + 1)
                for hh in months2:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.endTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.endTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.endTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)
                    df = xr.concat([df, ds[variable]], dim='time')
                    counter = counter + 1

            else:
                months1 = np.arange(self.startTime[1], 13)
                counter = 0
                for hh in months1:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.startTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.startTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.startTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)

                    if counter == 0:
                        df = ds[variable]
                    else:
                        df = xr.concat([df, ds[variable]], dim='time')
                        counter = counter + 1
                months2 = np.arange(1, 13)
                years = np.arange(self.startTime[0] + 1, self.endTime[0])
                for ff in years:
                    for hh in months2:
                        dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(ff) + str(hh).zfill(2) + '.nc'
                        ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(ff), dataLoc)
                        print('downloading {}-{}'.format(ff, hh))
                        ds = xr.open_dataset(ncfileURL)
                        df = xr.concat([df, ds[variable]], dim='time')
                        counter = counter + 1

                months3 = np.arange(1, self.endTime[1] + 1)
                for hh in months3:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.endTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.endTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.endTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)
                    df = xr.concat([df, ds[variable]], dim='time')
                    counter = counter + 1

            t2 = time.time()

            elapsed = t2 - t1
            print('Partial WIS data files took:')
            str(timedelta(seconds=elapsed))

        else:
            t1 = time.time()
            if self.startTime[0] == self.endTime[0]:
                months = np.arange(self.startTime[1], self.endTime[1] + 1)
                counter = 0
                for hh in months:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.startTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.startTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.startTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)

                    if counter == 0:
                        df = ds
                    else:
                        df = xr.concat([df, ds], dim='time')
                        counter = counter + 1

            elif self.startTime[0] == (self.endTime[0] - 1):
                months1 = np.arange(self.startTime[1], 13)
                counter = 0
                for hh in months1:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.startTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.startTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.startTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)

                    if counter == 0:
                        df = ds
                    else:
                        df = xr.concat([df, ds], dim='time')
                        counter = counter + 1
                months2 = np.arange(1, self.endTime[1] + 1)
                for hh in months2:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.endTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.endTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.endTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)
                    df = xr.concat([df, ds], dim='time')
                    counter = counter + 1

            else:
                months1 = np.arange(self.startTime[1], 13)
                counter = 0
                for hh in months1:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.startTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.startTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.startTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)

                    if counter == 0:
                        df = ds
                    else:
                        df = xr.concat([df, ds], dim='time')
                        counter = counter + 1
                months2 = np.arange(1, 13)
                years = np.arange(self.startTime[0] + 1, self.endTime[0])
                for ff in years:
                    for hh in months2:
                        dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(ff) + str(hh).zfill(2) + '.nc'
                        ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(ff), dataLoc)
                        print('downloading {}-{}'.format(ff, hh))
                        ds = xr.open_dataset(ncfileURL)
                        df = xr.concat([df, ds], dim='time')
                        counter = counter + 1

                months3 = np.arange(1, self.endTime[1] + 1)
                for hh in months3:
                    dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(self.endTime[0]) + str(hh).zfill(2) + '.nc'
                    ncfileURL = urljoin(self.chlDataLoc, basin, buoy, str(self.endTime[0]), dataLoc)
                    print('downloading {}-{}'.format(self.endTime[0], hh))
                    ds = xr.open_dataset(ncfileURL)
                    df = xr.concat([df, ds], dim='time')
                    counter = counter + 1

            t2 = time.time()

            elapsed = t2 - t1
            print('Full WIS data files took:')
            str(timedelta(seconds=elapsed))

        Hs = np.array(df.waveHs.values)
        Tp = np.array(df.waveTpPeak.values)
        Dm = np.array(df.waveMeanDirection.values)
        timeWave = df.time.values

        tC = np.array([datetime.utcfromtimestamp((x - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')) for x in timeWave])



        #
        # Hs = []
        # Tp = []
        # Dm = []
        # timeWave = []
        # for i in waveFiles_path:
        #     waves = loadWIS(i)
        #     Hs = np.append(Hs, waves['waveHs'])
        #     Tp = np.append(Tp, waves['waveTp'])
        #     Dm = np.append(Dm, waves['waveMeanDirection'])
        #     timeWave = np.append(timeWave, waves['t'].flatten())
        #
        # # tWave = [DT.datetime.fromtimestamp(x) for x in timeWave]
        # tWave = [datetime.fromtimestamp(x) for x in timeWave]
        # tC = np.array(tWave)
        # reorienting wave directions to FRF's shore normal (72 degrees)
        waveNorm = np.asarray(Dm) - int(self.shoreNormal)
        neg = np.where((waveNorm > 180))
        waveNorm[neg[0]] = waveNorm[neg[0]] - 360
        offpos = np.where((waveNorm > 90))
        offneg = np.where((waveNorm < -90))
        waveNorm[offpos[0]] = waveNorm[offpos[0]] * 0
        waveNorm[offneg[0]] = waveNorm[offneg[0]] * 0

        self.timeWave = tC
        self.Hs = Hs
        self.Tp = Tp
        self.Dm = waveNorm



    def downloadWIS(self,nodeID='ST84065',localDir='/volumes/anderson/WIS84065/'):
        from functions import get_elements
        from urllib.request import urlretrieve

        server_url = 'https://chldata.erdc.dren.mil/thredds/'
        request_url = os.path.join('catalog/wis/Pacific/',nodeID)

        years = np.arange(1980, 2024)
        for year in years:
            url = server_url + request_url + str(year) + '/catalog.xml'
            print(url)
            catalog = get_elements(url, 'dataset', 'urlPath')
            files = []
            for citem in catalog:
                if (citem[-3:] == '.nc'):
                    files.append(citem)
            count = 0

            file_subset = files  # [0:12]

            for f in file_subset:
                count += 1
                file_url = server_url + 'fileServer/' + f
                file_prefix = file_url.split('/')[-1][:-3]
                file_name = file_prefix + '.nc'
                # file_name = file_prefix + '_' + str(count) + '.nc'

                print('Downloaing file %d of %d' % (count, len(file_subset)))
                print(file_url)
                print(file_name)
                a = urlretrieve(file_url, localDir + file_name)
                print(a)

        return catalog, files, file_subset

    def getERA5WavesAndWinds(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta


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
            download_flag = True
            # api parameters
            params = {
                "product_type": ["reanalysis"],
                "variable": ['significant_height_of_combined_wind_waves_and_swell','mean_wave_period','mean_wave_direction','10m_u_component_of_wind','10m_v_component_of_wind'],
                'year': [str(extractTime[hh].year)],
                'month': [str(extractTime[hh].month).zfill(2), ],
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
                "data_format": "grib",
                "download_format": "unarchived",
                "area": [self.latTop, self.lonLeft, self.latBot, self.lonRight],
            }
            # retrieves the path to the file
            fl = cds.retrieve(dataset, params,'./output.grib')
            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())
            f = "./output.grib"
            ds = xr.open_dataset(f, engine='cfgrib')
            m, n, p = np.shape(ds.swh)
            SWH = np.zeros((n * p, m))
            MWP = np.zeros((n * p, m))
            MWD = np.zeros((n * p, m))
            U10 = np.zeros((n * p, m))
            V10 = np.zeros((n * p, m))


            for mmm in range(m):
                SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.time.values
                Dm = MWD
                Tp = MWP
                Hs = SWH
                u10 = U10
                v10 = V10

            else:
                tC = np.hstack((tC,ds.time.values))
                Dm = np.hstack((Dm,MWD))
                Tp = np.hstack((Tp,MWP))
                Hs = np.hstack((Hs,SWH))
                u10 = np.hstack((u10,U10))
                v10 = np.hstack((v10,V10))
            counter = counter + 1

        print('Extracted until {}-{}'.format(year2, month2))

        # waveNorm = np.asarray(Dm.flatten()) - int(self.shoreNormal)
        # neg = np.where((waveNorm > 180))
        # waveNorm[neg[0]] = waveNorm[neg[0]] - 360
        # offpos = np.where((waveNorm > 90))
        # offneg = np.where((waveNorm < -90))
        # waveNorm[offpos[0]] = waveNorm[offpos[0]] * 0
        # waveNorm[offneg[0]] = waveNorm[offneg[0]] * 0

        self.timeWave = tC
        self.Hs = Hs.flatten()
        self.Tp = Tp.flatten()
        self.Dm = Dm.flatten()
        self.timeWind = tC
        self.u10 = u10.flatten()
        self.v10 = v10.flatten()



    def getERA5Winds(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

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
                "variable": ['10m_u_component_of_wind','10m_v_component_of_wind'],
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

            m, n, p = np.shape(ds.u10)
            U10 = np.zeros((n * p, m))
            V10 = np.zeros((n * p, m))

            for mmm in range(m):
                U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.time.values
                u10 = U10
                v10 = V10
            else:
                tC = np.hstack((tC,ds.time.values))
                u10 = np.hstack((u10,U10))
                v10 = np.hstack((v10,V10))
            counter = counter + 1

        print('Extracted until {}-{}'.format(year2, month2))

        # waveNorm = np.asarray(Dm.flatten()) - int(self.shoreNormal)
        # neg = np.where((waveNorm > 180))
        # waveNorm[neg[0]] = waveNorm[neg[0]] - 360
        # offpos = np.where((waveNorm > 90))
        # offneg = np.where((waveNorm < -90))
        # waveNorm[offpos[0]] = waveNorm[offpos[0]] * 0
        # waveNorm[offneg[0]] = waveNorm[offneg[0]] * 0

        self.timeWind = tC
        self.u10 = u10.flatten()
        self.v10 = v10.flatten()




    def getERA5Waves(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

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

        self.extractTime = extractTime

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
            # # flag to download data
            # download_flag = False
            # api parameters
            params = {
                "product_type": ["reanalysis"],
                "variable": ['significant_height_of_combined_wind_waves_and_swell','mean_wave_period','mean_wave_direction'],
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
                "download_format": "unarchived",
                "data_format": "netcdf",

            }
            # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            cds.retrieve(dataset, params,os.path.join(self.savePath,"waves{}_{}.nc".format(extractTime[hh].year,extractTime[hh].month)))

            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())
            # with open("./output.nc", "rb") as f:
            #     ds = xr.open_dataset(f)
            ds = xr.open_dataset(os.path.join(self.savePath,"waves{}_{}.nc".format(extractTime[hh].year,extractTime[hh].month)))#,engine='netcdf4')

            m, n, p = np.shape(ds.swh)
            SWH = np.zeros((n * p, m))
            MWP = np.zeros((n * p, m))
            MWD = np.zeros((n * p, m))

            for mmm in range(m):
                SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.valid_time.values
                Dm = MWD
                Tp = MWP
                Hs = SWH
            else:
                tC = np.hstack((tC,ds.valid_time.values))
                Dm = np.hstack((Dm,MWD))
                Tp = np.hstack((Tp,MWP))
                Hs = np.hstack((Hs,SWH))
            counter = counter + 1

        print('Extracted until {}-{}'.format(year2, month2))
        print('Final time in file: {}'.format(ds.valid_time.values[-1]))

        # import os
        # if os.path.exists("./output.nc"):
        #     os.remove("./output.nc")
        # else:
        #     print("The file does not exist")
        del cds
        del ds
        # waveNorm = np.asarray(Dm.flatten()) - int(self.shoreNormal)
        # neg = np.where((waveNorm > 180))
        # waveNorm[neg[0]] = waveNorm[neg[0]] - 360
        # offpos = np.where((waveNorm > 90))
        # offneg = np.where((waveNorm < -90))
        # waveNorm[offpos[0]] = waveNorm[offpos[0]] * 0
        # waveNorm[offneg[0]] = waveNorm[offneg[0]] * 0

        self.timeWave = tC
        self.Hs = Hs.flatten()
        self.Tp = Tp.flatten()
        self.Dm = Dm.flatten()





    def getERA5WavesAndWindsAndTemps(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

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
                "variable": ['significant_height_of_combined_wind_waves_and_swell','mean_wave_period','mean_wave_direction','10m_u_component_of_wind','10m_v_component_of_wind', '2m_temperature','sea_surface_temperature', 'surface_net_solar_radiation'],
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

            m, n, p = np.shape(ds.swh)
            SWH = np.zeros((n * p, m))
            MWP = np.zeros((n * p, m))
            MWD = np.zeros((n * p, m))
            U10 = np.zeros((n * p, m))
            V10 = np.zeros((n * p, m))
            T2M = np.zeros((n * p, m))
            SST = np.zeros((n * p, m))
            SSR = np.zeros((n * p, m))

            for mmm in range(m):
                SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()
                T2M[:,mmm] = ds.t2m[mmm, :, :].values.flatten()
                SST[:,mmm] = ds.sst[mmm, :, :].values.flatten()
                SSR[:,mmm] = ds.ssr[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.time.values
                Dm = MWD
                Tp = MWP
                Hs = SWH
                u10 = U10
                v10 = V10
                t2m = T2M
                sst = SST
                ssr = SSR

            else:
                tC = np.hstack((tC,ds.time.values))
                Dm = np.hstack((Dm,MWD))
                Tp = np.hstack((Tp,MWP))
                Hs = np.hstack((Hs,SWH))
                u10 = np.hstack((u10,U10))
                v10 = np.hstack((v10,V10))
                t2m = np.hstack((t2m,T2M))
                sst = np.hstack((sst,SST))
                ssr = np.hstack((ssr,SSR))

            counter = counter + 1

        print('Extracted until {}-{}'.format(year2, month2))

        self.timeWave = tC
        self.Hs = Hs.flatten()
        self.Tp = Tp.flatten()
        self.Dm = Dm.flatten()
        self.timeWind = tC
        self.u10 = u10.flatten()
        self.v10 = v10.flatten()
        self.t2m = t2m.flatten()
        self.sst = sst.flatten()
        self.ssr = ssr.flatten()



    def getERA5Bathymetry(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta


        print('Extracting bathy')

        extractTime = date(self.startTime[0], self.startTime[1], self.startTime[2])

        import cdsapi
        import xarray as xr
        from urllib.request import urlopen

        if printToScreen == True:
            print('{}-{}'.format(extractTime.year, extractTime.month))
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
            "variable": ['model_bathymetry'],
            'year': [str(extractTime.year)],
            'month': [str(extractTime.month), ],
            'day': ['01'],
            "time": ['00:00'],
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

        m, n, p = np.shape(ds.wmb)
        WMB = np.zeros((n * p, m))

        for mmm in range(m):
            WMB[:,mmm] = ds.wmb[mmm, :, :].values.flatten()

        self.wmb = WMB.flatten()






    def getYearlyERA5Waves(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        dt = date(self.startTime[0], self.startTime[1], self.startTime[2])
        end = date(self.endTime[0], self.endTime[1], self.endTime[2])
        step = relativedelta(years=1)
        extractTime = []
        while dt < end:
            extractTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        print(extractTime)

        import cdsapi
        import xarray as xr
        from urllib.request import urlopen

        counter = 0
        for hh in range(len(extractTime)):

            if extractTime[hh].year == self.endTime[0]:
                for ttt in range(self.endTime[1]):
                    if printToScreen == True:
                        print('Now at the monthly scale: {}-{}'.format(extractTime[hh].year, ttt+1))

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
                        "variable": ['significant_height_of_combined_wind_waves_and_swell', 'mean_wave_period'],
                                    #'mean_wave_direction', '10m_u_component_of_wind', '10m_v_component_of_wind',
                                    #'2m_temperature', 'sea_surface_temperature', 'surface_net_solar_radiation'],
                        'year': [str(self.endTime[0])],
                        'month': [str(ttt+1), ],
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
                    # # retrieves the path to the file
                    # fl = cds.retrieve(dataset, params)
                    # # download the file
                    # if download_flag:
                    #     fl.download("./output.nc")
                    # # load into memory
                    # with urlopen(fl.location) as f:
                    #     ds = xr.open_dataset(f.read())

                    # retrieves the path to the file
                    # fl = cds.retrieve(dataset, params)
                    cds.retrieve(dataset, params, os.path.join(self.savePath, "wavesYearly{}_{}".format(extractTime[hh].year,
                                                                                                  extractTime[
                                                                                                      hh].month)))

                    # # download the file
                    # if download_flag:
                    #     fl.download("./output.nc")
                    # load into memory
                    # with urlopen(fl.location) as f:
                    #     ds = xr.open_dataset(f.read())
                    # with open("./output.nc", "rb") as f:
                    #     ds = xr.open_dataset(f)
                    ds = xr.open_dataset(os.path.join(self.savePath, "wavesYearly{}_{}".format(extractTime[hh].year,
                                                                                         extractTime[
                                                                                             hh].month)))  # ,engine='netcdf4')


                    m, n, p = np.shape(ds.swh)
                    SWH = np.zeros((n * p, m))
                    MWP = np.zeros((n * p, m))
                    MWD = np.zeros((n * p, m))
                    # U10 = np.zeros((n * p, m))
                    # V10 = np.zeros((n * p, m))
                    # T2M = np.zeros((n * p, m))
                    # SST = np.zeros((n * p, m))
                    # SSR = np.zeros((n * p, m))

                    for mmm in range(m):
                        SWH[:, mmm] = ds.swh[mmm, :, :].values.flatten()
                        MWP[:, mmm] = ds.mwp[mmm, :, :].values.flatten()
                        MWD[:, mmm] = ds.mwd[mmm, :, :].values.flatten()
                        # U10[:, mmm] = ds.u10[mmm, :, :].values.flatten()
                        # V10[:, mmm] = ds.v10[mmm, :, :].values.flatten()
                        # T2M[:, mmm] = ds.t2m[mmm, :, :].values.flatten()
                        # SST[:, mmm] = ds.sst[mmm, :, :].values.flatten()
                        # SSR[:, mmm] = ds.ssr[mmm, :, :].values.flatten()

                    if counter == 0:
                        tC = ds.valid_time.values
                        Dm = MWD
                        Tp = MWP
                        Hs = SWH
                        # u10 = U10
                        # v10 = V10
                        # t2m = T2M
                        # sst = SST
                        # ssr = SSR

                    else:
                        tC = np.hstack((tC, ds.valid_time.values))
                        Dm = np.hstack((Dm, MWD))
                        Tp = np.hstack((Tp, MWP))
                        Hs = np.hstack((Hs, SWH))
                        # u10 = np.hstack((u10, U10))
                        # v10 = np.hstack((v10, V10))
                        # t2m = np.hstack((t2m, T2M))
                        # sst = np.hstack((sst, SST))
                        # ssr = np.hstack((ssr, SSR))

                    counter = counter + 1

                # print('Extracted until {}-{}'.format(year2, month2))


            else:
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
                    "variable": ['significant_height_of_combined_wind_waves_and_swell','mean_wave_period','mean_wave_direction'],#'10m_u_component_of_wind','10m_v_component_of_wind', '2m_temperature','sea_surface_temperature', 'surface_net_solar_radiation'],
                    'year': [str(extractTime[hh].year)],
                    'month': ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12',],
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
                # # retrieves the path to the file
                # fl = cds.retrieve(dataset, params)
                # # download the file
                # if download_flag:
                #     fl.download("./output.nc")
                # # load into memory
                # with urlopen(fl.location) as f:
                #     ds = xr.open_dataset(f.read())

                # retrieves the path to the file
                # fl = cds.retrieve(dataset, params)
                cds.retrieve(dataset, params, os.path.join(self.savePath, "wavesYearly{}_{}".format(extractTime[hh].year,
                                                                                              extractTime[hh].month)))

                # # download the file
                # if download_flag:
                #     fl.download("./output.nc")
                # load into memory
                # with urlopen(fl.location) as f:
                #     ds = xr.open_dataset(f.read())
                # with open("./output.nc", "rb") as f:
                #     ds = xr.open_dataset(f)
                ds = xr.open_dataset(os.path.join(self.savePath, "wavesYearly{}_{}".format(extractTime[hh].year, extractTime[
                    hh].month)))  # ,engine='netcdf4')

                m, n, p = np.shape(ds.swh)
                SWH = np.zeros((n * p, m))
                MWP = np.zeros((n * p, m))
                MWD = np.zeros((n * p, m))
                # U10 = np.zeros((n * p, m))
                # V10 = np.zeros((n * p, m))
                # T2M = np.zeros((n * p, m))
                # SST = np.zeros((n * p, m))
                # SSR = np.zeros((n * p, m))

                for mmm in range(m):
                    SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                    MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                    MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                    # U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                    # V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()
                    # T2M[:,mmm] = ds.t2m[mmm, :, :].values.flatten()
                    # SST[:,mmm] = ds.sst[mmm, :, :].values.flatten()
                    # SSR[:,mmm] = ds.ssr[mmm, :, :].values.flatten()

                if counter == 0:
                    tC = ds.valid_time.values
                    Dm = MWD
                    Tp = MWP
                    Hs = SWH
                    # u10 = U10
                    # v10 = V10
                    # t2m = T2M
                    # sst = SST
                    # ssr = SSR

                else:
                    tC = np.hstack((tC,ds.valid_time.values))
                    Dm = np.hstack((Dm,MWD))
                    Tp = np.hstack((Tp,MWP))
                    Hs = np.hstack((Hs,SWH))
                    # u10 = np.hstack((u10,U10))
                    # v10 = np.hstack((v10,V10))
                    # t2m = np.hstack((t2m,T2M))
                    # sst = np.hstack((sst,SST))
                    # ssr = np.hstack((ssr,SSR))

                counter = counter + 1

            # print('Extracted until {}-{}'.format(year2, month2))

        self.timeWave = tC
        self.Hs = Hs.flatten()
        self.Tp = Tp.flatten()
        self.Dm = Dm.flatten()
        # self.timeWind = tC
        # self.u10 = u10.flatten()
        # self.v10 = v10.flatten()
        # self.t2m = t2m.flatten()
        # self.sst = sst.flatten()
        # self.ssr = ssr.flatten()


    def getBiMonthlyERA5Waves(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        dt = date(self.startTime[0], self.startTime[1], self.startTime[2])
        end = date(self.endTime[0], self.endTime[1], self.endTime[2])
        step = relativedelta(months=2)
        extractTime = []
        while dt < end:
            extractTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        print(extractTime)

        import cdsapi
        import xarray as xr
        from urllib.request import urlopen

        counter = 0
        for hh in range(len(extractTime)):

            if extractTime[hh].year == self.endTime[0]:
                for ttt in range(self.endTime[1]):
                    if printToScreen == True:
                        print('Now at the monthly scale: {}-{}'.format(extractTime[hh].year, ttt+1))

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
                        "variable": ['significant_height_of_combined_wind_waves_and_swell', 'mean_wave_period'],
                                    #'mean_wave_direction', '10m_u_component_of_wind', '10m_v_component_of_wind',
                                    #'2m_temperature', 'sea_surface_temperature', 'surface_net_solar_radiation'],
                        'year': [str(self.endTime[0])],
                        'month': [str(ttt+1), ],
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
                    # # retrieves the path to the file
                    # fl = cds.retrieve(dataset, params)
                    # # download the file
                    # if download_flag:
                    #     fl.download("./output.nc")
                    # # load into memory
                    # with urlopen(fl.location) as f:
                    #     ds = xr.open_dataset(f.read())

                    # retrieves the path to the file
                    # fl = cds.retrieve(dataset, params)
                    cds.retrieve(dataset, params, os.path.join(self.savePath, "waves{}_{}.nc".format(extractTime[hh].year,
                                                                                                  extractTime[
                                                                                                      hh].month)))

                    # # download the file
                    # if download_flag:
                    #     fl.download("./output.nc")
                    # load into memory
                    # with urlopen(fl.location) as f:
                    #     ds = xr.open_dataset(f.read())
                    # with open("./output.nc", "rb") as f:
                    #     ds = xr.open_dataset(f)
                    ds = xr.open_dataset(os.path.join(self.savePath, "waves{}_{}.nc".format(extractTime[hh].year,
                                                                                         extractTime[
                                                                                             hh].month)))  # ,engine='netcdf4')


                    m, n, p = np.shape(ds.swh)
                    SWH = np.zeros((n * p, m))
                    MWP = np.zeros((n * p, m))
                    MWD = np.zeros((n * p, m))
                    # U10 = np.zeros((n * p, m))
                    # V10 = np.zeros((n * p, m))
                    # T2M = np.zeros((n * p, m))
                    # SST = np.zeros((n * p, m))
                    # SSR = np.zeros((n * p, m))

                    for mmm in range(m):
                        SWH[:, mmm] = ds.swh[mmm, :, :].values.flatten()
                        MWP[:, mmm] = ds.mwp[mmm, :, :].values.flatten()
                        MWD[:, mmm] = ds.mwd[mmm, :, :].values.flatten()
                        # U10[:, mmm] = ds.u10[mmm, :, :].values.flatten()
                        # V10[:, mmm] = ds.v10[mmm, :, :].values.flatten()
                        # T2M[:, mmm] = ds.t2m[mmm, :, :].values.flatten()
                        # SST[:, mmm] = ds.sst[mmm, :, :].values.flatten()
                        # SSR[:, mmm] = ds.ssr[mmm, :, :].values.flatten()

                    if counter == 0:
                        tC = ds.valid_time.values
                        Dm = MWD
                        Tp = MWP
                        Hs = SWH
                        # u10 = U10
                        # v10 = V10
                        # t2m = T2M
                        # sst = SST
                        # ssr = SSR

                    else:
                        tC = np.hstack((tC, ds.valid_time.values))
                        Dm = np.hstack((Dm, MWD))
                        Tp = np.hstack((Tp, MWP))
                        Hs = np.hstack((Hs, SWH))
                        # u10 = np.hstack((u10, U10))
                        # v10 = np.hstack((v10, V10))
                        # t2m = np.hstack((t2m, T2M))
                        # sst = np.hstack((sst, SST))
                        # ssr = np.hstack((ssr, SSR))

                    counter = counter + 1

                # print('Extracted until {}-{}'.format(year2, month2))


            else:
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
                    "variable": ['significant_height_of_combined_wind_waves_and_swell','mean_wave_period','mean_wave_direction'],#'10m_u_component_of_wind','10m_v_component_of_wind', '2m_temperature','sea_surface_temperature', 'surface_net_solar_radiation'],
                    'year': [str(extractTime[hh].year)],
                    # 'month': ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12',],
                    'month': [str(extractTime[hh].month + 0),str(extractTime[hh].month + 1)],
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
                # # retrieves the path to the file
                # fl = cds.retrieve(dataset, params)
                # # download the file
                # if download_flag:
                #     fl.download("./output.nc")
                # # load into memory
                # with urlopen(fl.location) as f:
                #     ds = xr.open_dataset(f.read())

                # retrieves the path to the file
                # fl = cds.retrieve(dataset, params)
                cds.retrieve(dataset, params, os.path.join(self.savePath, "waves{}_{}.nc".format(extractTime[hh].year,
                                                                                              extractTime[hh].month)))

                # # download the file
                # if download_flag:
                #     fl.download("./output.nc")
                # load into memory
                # with urlopen(fl.location) as f:
                #     ds = xr.open_dataset(f.read())
                # with open("./output.nc", "rb") as f:
                #     ds = xr.open_dataset(f)
                ds = xr.open_dataset(os.path.join(self.savePath, "waves{}_{}.nc".format(extractTime[hh].year, extractTime[
                    hh].month)))  # ,engine='netcdf4')

                m, n, p = np.shape(ds.swh)
                SWH = np.zeros((n * p, m))
                MWP = np.zeros((n * p, m))
                MWD = np.zeros((n * p, m))
                # U10 = np.zeros((n * p, m))
                # V10 = np.zeros((n * p, m))
                # T2M = np.zeros((n * p, m))
                # SST = np.zeros((n * p, m))
                # SSR = np.zeros((n * p, m))

                for mmm in range(m):
                    SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                    MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                    MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                    # U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                    # V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()
                    # T2M[:,mmm] = ds.t2m[mmm, :, :].values.flatten()
                    # SST[:,mmm] = ds.sst[mmm, :, :].values.flatten()
                    # SSR[:,mmm] = ds.ssr[mmm, :, :].values.flatten()

                if counter == 0:
                    tC = ds.valid_time.values
                    Dm = MWD
                    Tp = MWP
                    Hs = SWH
                    # u10 = U10
                    # v10 = V10
                    # t2m = T2M
                    # sst = SST
                    # ssr = SSR

                else:
                    tC = np.hstack((tC,ds.valid_time.values))
                    Dm = np.hstack((Dm,MWD))
                    Tp = np.hstack((Tp,MWP))
                    Hs = np.hstack((Hs,SWH))
                    # u10 = np.hstack((u10,U10))
                    # v10 = np.hstack((v10,V10))
                    # t2m = np.hstack((t2m,T2M))
                    # sst = np.hstack((sst,SST))
                    # ssr = np.hstack((ssr,SSR))

                counter = counter + 1

            # print('Extracted until {}-{}'.format(year2, month2))

        self.timeWave = tC
        self.Hs = Hs.flatten()
        self.Tp = Tp.flatten()
        self.Dm = Dm.flatten()
        # self.timeWind = tC
        # self.u10 = u10.flatten()
        # self.v10 = v10.flatten()
        # self.t2m = t2m.flatten()
        # self.sst = sst.flatten()
        # self.ssr = ssr.flatten()

    def getTriMonthlyERA5Waves(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        dt = date(self.startTime[0], self.startTime[1], self.startTime[2])
        end = date(self.endTime[0], self.endTime[1], self.endTime[2])
        step = relativedelta(months=3)
        extractTime = []
        while dt < end:
            extractTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        print(extractTime)

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
                "variable": ['significant_height_of_combined_wind_waves_and_swell','mean_wave_period','mean_wave_direction'],#'10m_u_component_of_wind','10m_v_component_of_wind', '2m_temperature','sea_surface_temperature', 'surface_net_solar_radiation'],
                'year': [str(extractTime[hh].year)],
                # 'month': ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12',],
                'month': [str(extractTime[hh].month + 0),str(extractTime[hh].month + 1),str(extractTime[hh].month + 2)],
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
            # # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())

            # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            cds.retrieve(dataset, params, os.path.join(self.savePath, "waves{}_{}.nc".format(extractTime[hh].year,
                                                                                             extractTime[hh].month)))

            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())
            # with open("./output.nc", "rb") as f:
            #     ds = xr.open_dataset(f)
            ds = xr.open_dataset(os.path.join(self.savePath, "waves{}_{}.nc".format(extractTime[hh].year, extractTime[
                hh].month)))  # ,engine='netcdf4')

            m, n, p = np.shape(ds.swh)
            SWH = np.zeros((n * p, m))
            MWP = np.zeros((n * p, m))
            MWD = np.zeros((n * p, m))
            # U10 = np.zeros((n * p, m))
            # V10 = np.zeros((n * p, m))
            # T2M = np.zeros((n * p, m))
            # SST = np.zeros((n * p, m))
            # SSR = np.zeros((n * p, m))

            for mmm in range(m):
                SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                # U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                # V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()
                # T2M[:,mmm] = ds.t2m[mmm, :, :].values.flatten()
                # SST[:,mmm] = ds.sst[mmm, :, :].values.flatten()
                # SSR[:,mmm] = ds.ssr[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.valid_time.values
                Dm = MWD
                Tp = MWP
                Hs = SWH
                # u10 = U10
                # v10 = V10
                # t2m = T2M
                # sst = SST
                # ssr = SSR

            else:
                tC = np.hstack((tC,ds.valid_time.values))
                Dm = np.hstack((Dm,MWD))
                Tp = np.hstack((Tp,MWP))
                Hs = np.hstack((Hs,SWH))
                # u10 = np.hstack((u10,U10))
                # v10 = np.hstack((v10,V10))
                # t2m = np.hstack((t2m,T2M))
                # sst = np.hstack((sst,SST))
                # ssr = np.hstack((ssr,SSR))

            counter = counter + 1

        # print('Extracted until {}-{}'.format(year2, month2))

        self.timeWave = tC
        self.Hs = Hs.flatten()
        self.Tp = Tp.flatten()
        self.Dm = Dm.flatten()
        # self.timeWind = tC
        # self.u10 = u10.flatten()
        # self.v10 = v10.flatten()
        # self.t2m = t2m.flatten()
        # self.sst = sst.flatten()
        # self.ssr = ssr.flatten()


    def getTriMonthlyERA5Atms(self,printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        dt = date(self.startTime[0], self.startTime[1], self.startTime[2])
        end = date(self.endTime[0], self.endTime[1], self.endTime[2])
        step = relativedelta(months=3)
        extractTime = []
        while dt < end:
            extractTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        print(extractTime)

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
                "variable": ['10m_u_component_of_wind','10m_v_component_of_wind', '2m_temperature'],
                'year': [str(extractTime[hh].year)],
                # 'month': ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12',],
                'month': [str(extractTime[hh].month + 0),str(extractTime[hh].month + 1),str(extractTime[hh].month + 2)],
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
            # # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())

            # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            cds.retrieve(dataset, params, os.path.join(self.savePath, "winds{}_{}.nc".format(extractTime[hh].year,
                                                                                             extractTime[hh].month)))

            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())
            # with open("./output.nc", "rb") as f:
            #     ds = xr.open_dataset(f)
            ds = xr.open_dataset(os.path.join(self.savePath, "winds{}_{}.nc".format(extractTime[hh].year, extractTime[
                hh].month)))  # ,engine='netcdf4')

            m, n, p = np.shape(ds.u10)
            # SWH = np.zeros((n * p, m))
            # MWP = np.zeros((n * p, m))
            # MWD = np.zeros((n * p, m))
            U10 = np.zeros((n * p, m))
            V10 = np.zeros((n * p, m))
            T2M = np.zeros((n * p, m))
            # SST = np.zeros((n * p, m))
            # SSR = np.zeros((n * p, m))

            for mmm in range(m):
                # SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                # MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                # MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                U10[:,mmm] = ds.u10[mmm, :, :].values.flatten()
                V10[:,mmm] = ds.v10[mmm, :, :].values.flatten()
                T2M[:,mmm] = ds.t2m[mmm, :, :].values.flatten()
                # SST[:,mmm] = ds.sst[mmm, :, :].values.flatten()
                # SSR[:,mmm] = ds.ssr[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.valid_time.values
                # Dm = MWD
                # Tp = MWP
                # Hs = SWH
                u10 = U10
                v10 = V10
                t2m = T2M
                # sst = SST
                # ssr = SSR

            else:
                tC = np.hstack((tC,ds.valid_time.values))
                # Dm = np.hstack((Dm,MWD))
                # Tp = np.hstack((Tp,MWP))
                # Hs = np.hstack((Hs,SWH))
                u10 = np.hstack((u10,U10))
                v10 = np.hstack((v10,V10))
                t2m = np.hstack((t2m,T2M))
                # sst = np.hstack((sst,SST))
                # ssr = np.hstack((ssr,SSR))

            counter = counter + 1

            # print('Extracted until {}-{}'.format(year2, month2))

        self.timeWave = tC
        # self.Hs = Hs.flatten()
        # self.Tp = Tp.flatten()
        # self.Dm = Dm.flatten()
        # self.timeWind = tC
        self.u10 = u10.flatten()
        self.v10 = v10.flatten()
        self.t2m = t2m.flatten()
        # self.sst = sst.flatten()
        # self.ssr = ssr.flatten()

    def getTriMonthlyERA5Atms(self, printToScreen=False):
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        dt = date(self.startTime[0], self.startTime[1], self.startTime[2])
        end = date(self.endTime[0], self.endTime[1], self.endTime[2])
        step = relativedelta(months=3)
        extractTime = []
        while dt < end:
            extractTime.append(dt)  # .strftime('%Y-%m-%d'))
            dt += step

        print(extractTime)

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
                "variable": ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature'],
                'year': [str(extractTime[hh].year)],
                # 'month': ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12',],
                'month': [str(extractTime[hh].month + 0), str(extractTime[hh].month + 1),
                          str(extractTime[hh].month + 2)],
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
            # # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())

            # retrieves the path to the file
            # fl = cds.retrieve(dataset, params)
            cds.retrieve(dataset, params, os.path.join(self.savePath, "winds{}_{}.nc".format(extractTime[hh].year,
                                                                                             extractTime[hh].month)))

            # # download the file
            # if download_flag:
            #     fl.download("./output.nc")
            # load into memory
            # with urlopen(fl.location) as f:
            #     ds = xr.open_dataset(f.read())
            # with open("./output.nc", "rb") as f:
            #     ds = xr.open_dataset(f)
            ds = xr.open_dataset(os.path.join(self.savePath, "winds{}_{}.nc".format(extractTime[hh].year, extractTime[
                hh].month)))  # ,engine='netcdf4')

            m, n, p = np.shape(ds.u10)
            # SWH = np.zeros((n * p, m))
            # MWP = np.zeros((n * p, m))
            # MWD = np.zeros((n * p, m))
            U10 = np.zeros((n * p, m))
            V10 = np.zeros((n * p, m))
            T2M = np.zeros((n * p, m))
            # SST = np.zeros((n * p, m))
            # SSR = np.zeros((n * p, m))

            for mmm in range(m):
                # SWH[:,mmm] = ds.swh[mmm, :, :].values.flatten()
                # MWP[:,mmm] = ds.mwp[mmm, :, :].values.flatten()
                # MWD[:,mmm] = ds.mwd[mmm, :, :].values.flatten()
                U10[:, mmm] = ds.u10[mmm, :, :].values.flatten()
                V10[:, mmm] = ds.v10[mmm, :, :].values.flatten()
                T2M[:, mmm] = ds.t2m[mmm, :, :].values.flatten()
                # SST[:,mmm] = ds.sst[mmm, :, :].values.flatten()
                # SSR[:,mmm] = ds.ssr[mmm, :, :].values.flatten()

            if counter == 0:
                tC = ds.valid_time.values
                # Dm = MWD
                # Tp = MWP
                # Hs = SWH
                u10 = U10
                v10 = V10
                t2m = T2M
                # sst = SST
                # ssr = SSR

            else:
                tC = np.hstack((tC, ds.valid_time.values))
                # Dm = np.hstack((Dm,MWD))
                # Tp = np.hstack((Tp,MWP))
                # Hs = np.hstack((Hs,SWH))
                u10 = np.hstack((u10, U10))
                v10 = np.hstack((v10, V10))
                t2m = np.hstack((t2m, T2M))
                # sst = np.hstack((sst,SST))
                # ssr = np.hstack((ssr,SSR))

            counter = counter + 1

            # print('Extracted until {}-{}'.format(year2, month2))

        self.timeWave = tC
        # self.Hs = Hs.flatten()
        # self.Tp = Tp.flatten()
        # self.Dm = Dm.flatten()
        # self.timeWind = tC
        self.u10 = u10.flatten()
        self.v10 = v10.flatten()
        self.t2m = t2m.flatten()
        # self.sst = sst.flatten()
        # self.ssr = ssr.flatten()
