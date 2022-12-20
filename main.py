# This is a sample Python script.
from weatherTypes import weatherTypes
import plotting
from metOcean import getMetOcean

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    slpPath = '/users/dylananderson/documents/data/prmsl/'
    wisPath = '/users/dylananderson/documents/data/WIS_ST63218/'
    wlPath = '/users/dylananderson/documents/data/frfWaterLevel/'
    startTime = [1981, 1, 1]
    endTime = [1981, 12, 1]
    duckSLP = weatherTypes(slpPath=slpPath,startTime=startTime,endTime=endTime)
    duckSLP.extractCFSR()

    plotting.plotSlpExample(struct=duckSLP)
    duckSLP.pcaOfSlps()
    plotting.plotEOFs(struct=duckSLP)
    duckSLP.wtClusters(numClusters=9)
    plotting.plotWTs(struct=duckSLP)
    duckMET = getMetOcean(wlPath=wlPath,wisPath=wisPath,startTime=startTime,endTime=endTime)
    duckMET.getWIS()
    duckMET.getWaterLevels()
    plotting.plotOceanConditions(struct=duckMET)
