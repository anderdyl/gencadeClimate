# This is a sample Python script.
from weatherTypes import weatherTypes
import plotting
from metOcean import getMetOcean

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



slpPath = '/volumes/macDrive/prmsl/'
wisPath = '/volumes/macDrive/WIS63142/'
wlPath = '/users/dylananderson/documents/data/frfWaterLevel/'
startTime = [1979, 1, 1]
endTime = [2024, 5, 31]

from climateIndices import climateIndices

# test = climateIndices(awtStart=1880,awtEnd=2024)
# test.atlanticAWT(plotOutput=True)
# test.mjo(historicalSimNum=100,futureSimNum=100,loadPrevious=False,plotOutput=True)


# duckSLP = weatherTypes(slpPath=slpPath,startTime=startTime,endTime=endTime)
# duckSLP.extractCFSR(printToScreen=True)
# #
# # plotting.plotSlpExample(struct=duckSLP)
# duckSLP.pcaOfSlps()
# # plotting.plotEOFs(struct=duckSLP)

import pickle
with open(r"duckLatestCFSR.pickle", "rb") as input_file:
   priorComputations = pickle.load(input_file)
duckSLP = priorComputations['duckSlp']
duckMET = priorComputations['duckMet']
test = priorComputations['climate']

# duckSLP.wtClusters(numClusters=49,TCs=True,Basin=b'NA')
# plotting.plotWTs(struct=duckSLP)
# plotting.plotSeasonal(struct=duckSLP)
#
# # import pickle
# # outdict = {}
# # # outdict['duckMet'] = duckMET
# # outdict['duckSlp'] = duckSLP
# # outdict['endTime'] = endTime
# # outdict['startTime'] = startTime
# # with open('duckLatestCFSR.pickle', 'wb') as f:
# #     pickle.dump(outdict, f)
#
#
#
#
# duckMET = getMetOcean(wlPath=wlPath,wisPath=wisPath,startTime=startTime,endTime=endTime)
# duckMET.getWISLocal()
# # duckMET.getWIS()
# # #def getWISThredds(self,basin,buoy,**kwargs):
# # duckMET.getWISThredds(basin = 'Atlantic',buoy = 'ST63218',variables = ['waveHs','waveTpPeak','waveMeanDirection'])
# duckMET.getWaterLevels()
# plotting.plotOceanConditions(struct=duckMET)
#
# duckSLP.separateHistoricalHydrographs(metOcean=duckMET)
# duckSLP.metOceanCopulas()
# #

# duckSLP.alrSimulations(climate=test,historicalSimNum=10,futureSimNum=10)
# duckSLP.simsFutureInterpolated(simNum=10)

duckSLP.simsFutureValidated(met=duckMET)


# import pickle
# outdict = {}
# outdict['duckMet'] = duckMET
# outdict['duckSlp'] = duckSLP
# outdict['climate'] = test
#
# outdict['endTime'] = endTime
# outdict['startTime'] = startTime
# with open('duckLatestCFSR.pickle', 'wb') as f:
#     pickle.dump(outdict, f)



