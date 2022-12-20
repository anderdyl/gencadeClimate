
import numpy as np
import os
from functions import loadWIS, loadWaterLevel
from datetime import datetime, timedelta

class getMetOcean():
    '''
    Class to get metOcean Data
    '''

    def __init__(self, **kwargs):

        self.avgTime = kwargs.get('avgTime',24)
        self.startTime = kwargs.get('startTime',[1979,1,1])
        self.endTime = kwargs.get('endTime',[2020,12,31])
        self.wisPath = kwargs.get('wisPath','/users/dylananderson/documents/data/WIS_ST63218/')
        self.wlPath = kwargs.get('wlPath','/users/dylananderson/documents/data/frfWaterLevel/')
        self.shoreNormal = kwargs.get('shoreNormal',72)



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



    def getWIS(self):

        # LOADING THE WIS FILES
        # Need to sort the files to ensure correct temporal order...
        waveFiles = os.listdir(self.wisPath)
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
        timeWave = []
        for i in waveFiles_path:
            waves = loadWIS(i)
            Hs = np.append(Hs, waves['waveHs'])
            Tp = np.append(Tp, waves['waveTp'])
            Dm = np.append(Dm, waves['waveMeanDirection'])
            timeWave = np.append(timeWave, waves['t'].flatten())

        # tWave = [DT.datetime.fromtimestamp(x) for x in timeWave]
        tWave = [datetime.fromtimestamp(x) for x in timeWave]
        tC = np.array(tWave)
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
