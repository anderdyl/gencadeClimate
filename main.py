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
wisPath = '/volumes/macDrive/WIS_ST63218/'
wlPath = '/users/dylananderson/documents/data/frfWaterLevel/'
startTime = [1980, 1, 1]
endTime = [2023, 6, 1]

from climateIndices import climateIndices

# test = climateIndices(awtStart=1880,awtEnd=2023)
# test.atlanticAWT(plotOutput=True)
# test.mjo(historicalSimNum=100,futureSimNum=100,loadPrevious=False,plotOutput=True)

duckSLP = weatherTypes(slpPath=slpPath,startTime=startTime,endTime=endTime)
duckSLP.extractCFSR(printToScreen=True)
# duckSLP.extractERA5(printToScreen=True)

plotting.plotSlpExample(struct=duckSLP)
duckSLP.pcaOfSlps()
plotting.plotEOFs(struct=duckSLP)
# duckSLP.wtClusters(numClusters=12)
# plotting.plotWTs(struct=duckSLP)
#

# duckMET = getMetOcean(wlPath=wlPath,wisPath=wisPath,startTime=startTime,endTime=endTime)
# #duckMET.getWIS()
# #def getWISThredds(self,basin,buoy,**kwargs):
# duckMET.getWISThredds(basin = 'Atlantic',buoy = 'ST63218',variables = ['waveHs','waveTpPeak','waveMeanDirection'])
# duckMET.getWaterLevels()
# plotting.plotOceanConditions(struct=duckMET)

    # import pickle
    # outdict = {}
    # outdict['duckMet'] = duckMET
    # outdict['duckSlp'] = duckSLP
    # outdict['endTime'] = endTime
    # outdict['startTime'] = startTime
    # with open('duckFirstERA5.pickle', 'wb') as f:
    #     pickle.dump(outdict, f)