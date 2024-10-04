# This is a sample Python script.
from weatherTypes import weatherTypes
import plotting
from metOcean import getMetOcean
from climateIndices import climateIndices

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


savePath = '/volumes/macDrive/va/'
slpPath = '/volumes/macDrive/prmsl/'
wisPath = '/volumes/macDrive/WIS63142/'
wlPath = '/users/dylananderson/documents/data/frfWaterLevel/'
startTime = [1979, 1, 1]
endTime = [2024, 8, 31]

#
# climate = climateIndices(awtStart=1880,awtEnd=2024,savePath=savePath)
# climate.atlanticAWT(plotOutput=True)
# climate.mjo(historicalSimNum=100,futureSimNum=100,loadPrevious=False,plotOutput=True)
#
#
# wts = weatherTypes(slpPath=slpPath,startTime=startTime,endTime=endTime,savePath=savePath)
# wts.extractCFSR(printToScreen=True)
# #
# # plotting.plotSlpExample(struct=duckSLP)
# wts.pcaOfSlps()
# # plotting.plotEOFs(struct=duckSLP)
# wts.wtClusters(numClusters=64,minGroupSize=50,TCs=False,RG='seasonal',Basin=b'NA',alphaRG=0.2)
# # plotting.plotWTs(struct=wts,withTCs=False)
# # plotting.plotSeasonal(struct=wts)


import os
import pickle
with open(os.path.join(savePath,'latestData.pickle'), "rb") as input_file:
   priorComputations = pickle.load(input_file)
wts = priorComputations['wts']
# duckMET = priorComputations['duckMet']
climate = priorComputations['climate']

#
# #
# #
# #
# #
# metOcean = getMetOcean(wlPath=wlPath,wisPath=wisPath,startTime=startTime,endTime=endTime,shoreNormal=155)
# metOcean.getWISLocal()
# # metOcean.getWIS()
# # #def getWISThredds(self,basin,buoy,**kwargs):
# # metOcean.getWISThredds(basin = 'Atlantic',buoy = 'ST63218',variables = ['waveHs','waveTpPeak','waveMeanDirection'])
# metOcean.getWaterLevels()
# # plotting.plotOceanConditions(struct=metOcean)
# #
# wts.separateHistoricalHydrographs(metOcean=metOcean)
wts.metOceanCopulas()
# #
wts.futureSimStart = 2024
wts.futureSimEnd = 2124
# wts.alrSimulations(climate=climate,historicalSimNum=10,futureSimNum=10)
wts.simsFutureInterpolated(simNum=10)

wts.simsFutureValidated(met=metOcean)

# import os
# import pickle
# outdict = {}
# outdict['metOcean'] = metOcean
# outdict['wts'] = wts
# outdict['climate'] = climate
# outdict['endTime'] = endTime
# outdict['startTime'] = startTime
# with open(os.path.join(savePath,'latestData.pickle'), 'wb') as f:
#     pickle.dump(outdict, f)



