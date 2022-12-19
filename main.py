# This is a sample Python script.
from weatherTypes import weatherTypes
import plotting
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    slpPath = '/users/dylananderson/documents/data/prmsl/'

    test = weatherTypes(slpPath=slpPath,startTime=[1979,1,1],endTime=[1979,12,1])
    test.extractCFSR()
    plotting.plotSlpExample(struct=test)
    test.pcaOfSlps()
    plotting.plotEOFs(struct=test)
    test.wtClusters(numClusters=9)
    plotting.plotWTs(struct=test)
