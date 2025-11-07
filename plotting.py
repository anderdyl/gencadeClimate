


def plotOceanConditions(struct):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    plt.figure()

    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    ax1.plot(struct.timeWave, struct.Hs, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax1.set_ylabel('Hs (m)')

    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=1, colspan=1)
    ax2.plot(struct.timeWave, struct.Tp, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax2.set_ylabel('Tp (s)')

    ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1)
    ax3.plot(struct.timeWave, struct.Dm, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax3.set_ylabel('Dm')

    ax4 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
    ax4.plot(struct.timeWL, struct.waterLevel, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('wl (m)')


def plotWTs(struct,withTCs=True):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    if withTCs == True:
        dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters+struct.num_clustersTC))

        # plotting the EOF patterns
        fig2 = plt.figure(figsize=(10, 10))
        # gs1 = gridspec.GridSpec(int(np.ceil(np.sqrt(struct.numClustersETC))), int(np.ceil(np.sqrt(struct.numClustersETC))))
        gs1 = gridspec.GridSpec(int(10), int(7))

        gs1.update(wspace=0.00, hspace=0.00)  # set the spacing between axes.
        c1 = 0
        c2 = 0
        counter = 0
        plotIndx = 0
        plotIndy = 0
        for hh in range(struct.numClusters):
            # p1 = plt.subplot2grid((6,6),(c1,c2))
            ax = plt.subplot(gs1[hh])
            num = struct.kmaOrderETC[hh]

            spatialField = struct.Km_ETC[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
                struct.xFlat[~struct.isOnLandFlat])] / 100
            # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))

            if struct.basin == 'atlantic':
                X_in = struct.xFlat[~struct.isOnLandFlat]
                X_in_Checker = np.where(X_in < 180)
                X_in[X_in_Checker] = X_in[X_in_Checker] + 360

            else:
                X_in = struct.xFlat[~struct.isOnLandFlat]

            Y_in = struct.yFlat[~struct.isOnLandFlat]
            sea_nodes = []
            for qq in range(len(X_in)):
                sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

            rectField = np.ones((np.shape(struct.xGrid))) * np.nan
            for tt in range(len(sea_nodes)):
                rectField[sea_nodes[tt]] = spatialField[tt]

            clevels = np.arange(-32, 32, 1)
            # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
            #             resolution='c')
            if struct.basin == 'atlantic':
                m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop,
                            llcrnrlon=struct.lonLeft,
                            urcrnrlon=struct.lonRight + 360, lat_ts=10,
                            resolution='c')
            else:
                m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop,
                            llcrnrlon=struct.lonLeft,
                            urcrnrlon=struct.lonRight, lat_ts=10,
                            resolution='c')
            m.fillcontinents(color=dwtcolors[hh])
            if struct.basin == 'atlantic':
                xGridCheck = struct.xGrid
                indexChecker = np.where(xGridCheck < 180)
                xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
                cx, cy = m(xGridCheck, struct.yGrid)
            else:
                cx, cy = m(struct.xGrid, struct.yGrid)

            m.drawcoastlines()
            CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
            # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
            tx, ty = m(320, 10)
            ax.text(tx, ty, '{}'.format(struct.groupSizeETC[num]))

            c2 += 1
            if c2 == int(np.ceil(np.sqrt(struct.numClusters)) - 1):
                c1 += 1
                c2 = 0

            if plotIndx <= np.ceil(np.sqrt(struct.numClusters)):
                ax.xaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
            if plotIndy > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks([])
            counter = counter + 1
            if plotIndy <= np.ceil(np.sqrt(struct.numClusters)):
                plotIndy = plotIndy + 1
            else:
                plotIndy = 0
                plotIndx = plotIndx + 1

        for hh in range(struct.num_clustersTC):
            # p1 = plt.subplot2grid((6,6),(c1,c2))
            ax = plt.subplot(gs1[hh+49])
            num = hh#struct.kmaOrderTC[hh]

            spatialField = struct.Km_TC[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
                struct.xFlat[~struct.isOnLandFlat])] / 100
            # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))

            if struct.basin == 'atlantic':
                X_in = struct.xFlat[~struct.isOnLandFlat]
                X_in_Checker = np.where(X_in < 180)
                X_in[X_in_Checker] = X_in[X_in_Checker] + 360

            else:
                X_in = struct.xFlat[~struct.isOnLandFlat]

            Y_in = struct.yFlat[~struct.isOnLandFlat]
            sea_nodes = []
            for qq in range(len(X_in)):
                sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

            rectField = np.ones((np.shape(struct.xGrid))) * np.nan
            for tt in range(len(sea_nodes)):
                rectField[sea_nodes[tt]] = spatialField[tt]

            clevels = np.arange(-32, 32, 1)
            # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
            #             resolution='c')
            if struct.basin == 'atlantic':
                m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop,
                            llcrnrlon=struct.lonLeft,
                            urcrnrlon=struct.lonRight + 360, lat_ts=10,
                            resolution='c')
            else:
                m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop,
                            llcrnrlon=struct.lonLeft,
                            urcrnrlon=struct.lonRight, lat_ts=10,
                            resolution='c')
            m.fillcontinents(color=dwtcolors[hh+49])

            if struct.basin == 'atlantic':
                xGridCheck = struct.xGrid
                indexChecker = np.where(xGridCheck < 180)
                xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
                cx, cy = m(xGridCheck, struct.yGrid)
            else:
                cx, cy = m(struct.xGrid, struct.yGrid)

            m.drawcoastlines()
            CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
            # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
            tx, ty = m(320, 10)
            ax.text(tx, ty, '{}'.format(struct.groupSizeTC[num]))

            c2 += 1
            if c2 == int(np.ceil(np.sqrt(struct.numClusters)) - 1):
                c1 += 1
                c2 = 0

            if plotIndx <= np.ceil(np.sqrt(struct.numClusters)):
                ax.xaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
            if plotIndy > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks([])
            counter = counter + 1
            if plotIndy <= np.ceil(np.sqrt(struct.numClusters)):
                plotIndy = plotIndy + 1
            else:
                plotIndy = 0
                plotIndx = plotIndx + 1


    else:
        dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters))

        # plotting the EOF patterns
        fig2 = plt.figure(figsize=(10, 10))
        gs1 = gridspec.GridSpec(int(np.ceil(np.sqrt(struct.numClusters))), int(np.ceil(np.sqrt(struct.numClusters))))
        gs1.update(wspace=0.00, hspace=0.00)  # set the spacing between axes.
        c1 = 0
        c2 = 0
        counter = 0
        plotIndx = 0
        plotIndy = 0
        for hh in range(struct.numClusters):
            # p1 = plt.subplot2grid((6,6),(c1,c2))
            ax = plt.subplot(gs1[hh])
            num = struct.kmaOrderETC[hh]

            spatialField = struct.Km_ETC[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
                struct.xFlat[~struct.isOnLandFlat])] / 100
            # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))

            if struct.basin == 'atlantic':
                X_in = struct.xFlat[~struct.isOnLandFlat]
                X_in_Checker = np.where(X_in < 180)
                X_in[X_in_Checker] = X_in[X_in_Checker] + 360

            else:
                X_in = struct.xFlat[~struct.isOnLandFlat]

            Y_in = struct.yFlat[~struct.isOnLandFlat]
            sea_nodes = []
            for qq in range(len(X_in)):
                sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

            rectField = np.ones((np.shape(struct.xGrid))) * np.nan
            for tt in range(len(sea_nodes)):
                rectField[sea_nodes[tt]] = spatialField[tt]

            clevels = np.arange(-32, 32, 1)
            # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
            #             resolution='c')

            if struct.basin == 'atlantic':
                m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop,
                            llcrnrlon=struct.lonLeft,
                            urcrnrlon=struct.lonRight + 360, lat_ts=10,
                            resolution='c')
            else:
                m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop,
                            llcrnrlon=struct.lonLeft,
                            urcrnrlon=struct.lonRight, lat_ts=10,
                            resolution='c')

            m.fillcontinents(color=dwtcolors[hh])
            cx, cy = m(struct.xGrid, struct.yGrid)
            m.drawcoastlines()
            CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
            # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
            tx, ty = m(320, 10)
            ax.text(tx, ty, '{}'.format(struct.groupSizeETC[num]))

            c2 += 1
            if c2 == int(np.ceil(np.sqrt(struct.numClusters)) - 1):
                c1 += 1
                c2 = 0

            if plotIndx <= np.ceil(np.sqrt(struct.numClusters)):
                ax.xaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
            if plotIndy > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks([])
            counter = counter + 1
            if plotIndy <= np.ceil(np.sqrt(struct.numClusters)):
                plotIndy = plotIndy + 1
            else:
                plotIndy = 0
                plotIndx = plotIndx + 1


def plotSlpExample(struct, plotTime=0):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    fig = plt.figure(figsize=(10, 6))

    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    clevels = np.arange(940, 1080, 1)

    spatialField = struct.SLPS[:, plotTime] / 100  # SLPS[:, i] / 100  # - np.nanmean(SLP, axis=1) / 100

    if struct.basin == 'atlantic':
        X_in = struct.xFlat[~struct.isOnLandFlat]
        X_in_Checker = np.where(X_in < 180)
        X_in[X_in_Checker] = X_in[X_in_Checker] + 360

    else:
        X_in = struct.xFlat[~struct.isOnLandFlat]
    Y_in = struct.yFlat[~struct.isOnLandFlat]
    sea_nodes = []
    for qq in range(len(X_in)):
        sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

    rectField = np.ones((np.shape(struct.xGrid))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    if struct.basin == 'atlantic':
        m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft,
                    urcrnrlon=struct.lonRight+360, lat_ts=10,
                    resolution='c')
    else:
        m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft,
                    urcrnrlon=struct.lonRight, lat_ts=10,
                    resolution='c')
    if struct.basin == 'atlantic':
        xGridCheck = struct.xGrid
        indexChecker = np.where(xGridCheck < 180)
        xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
        cx, cy = m(xGridCheck, struct.yGrid)
    else:
        cx, cy = m(struct.xGrid, struct.yGrid)
    m.fillcontinents(color=[0.5, 0.5, 0.5])

    m.drawcoastlines()
    # m.bluemarble()
    # CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r, shading='gouraud')
    CS = m.contourf(cx.T, cy.T, rectField.T, clevels, vmin=975, vmax=1045, cmap=cm.RdBu_r)  # , shading='gouraud')

    tx, ty = m(320, -0)
    parallels = np.arange(0, 360, 10)
    m.drawparallels(parallels, labels=[True, True, True, False], textcolor='black')
    # ax.text(tx, ty, '{}'.format((group_size[num])))
    meridians = np.arange(0, 360, 20)
    m.drawmeridians(meridians, labels=[True, True, True, True], textcolor='black')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(CS, cax=cbar_ax)
    cbar.set_label('SLP (mbar)')




def plotSlpExampleLocal(struct, plotTime=0, centralNode=(0,0)):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    fig = plt.figure(figsize=(10, 6))

    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    clevels = np.arange(940, 1080, 1)

    spatialField = struct.SLPSLocal[:, plotTime] / 100  # SLPS[:, i] / 100  # - np.nanmean(SLP, axis=1) / 100

    if struct.basin == 'atlantic':
        X_in = struct.xFlatLocal#[~struct.isOnLandFlatLocal]
        X_in_Checker = np.where(X_in < 180)
        X_in[X_in_Checker] = X_in[X_in_Checker] + 360

    else:
        X_in = struct.xFlatLocal#[~struct.isOnLandFlatLocal]
    Y_in = struct.yFlatLocal#[~struct.isOnLandFlatLocal]
    sea_nodes = []
    for qq in range(len(X_in)):
        sea_nodes.append(np.where((struct.xGridLocal == X_in[qq]) & (struct.yGridLocal == Y_in[qq])))

    rectField = np.ones((np.shape(struct.xGridLocal))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    if struct.basin == 'atlantic':
        m = Basemap(projection='merc', llcrnrlat=centralNode[0]-4, urcrnrlat=centralNode[0]+4, llcrnrlon=centralNode[1]-4,
                    urcrnrlon=centralNode[1]+4, lat_ts=10,
                    resolution='l')
    else:
        m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft,
                    urcrnrlon=struct.lonRight, lat_ts=10,
                    resolution='l')
    if struct.basin == 'atlantic':
        xGridCheck = struct.xGridLocal
        indexChecker = np.where(xGridCheck < 180)
        xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
        cx, cy = m(xGridCheck, struct.yGridLocal)
    else:
        cx, cy = m(struct.xGridLocal, struct.yGridLocal)
    m.fillcontinents(color=[0.5, 0.5, 0.5])

    m.drawcoastlines()
    # m.bluemarble()
    # CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r, shading='gouraud')
    CS = m.contourf(cx.T, cy.T, rectField.T, clevels, vmin=975, vmax=1045, cmap=cm.RdBu_r)  # , shading='gouraud')

    tx, ty = m(320, -0)
    parallels = np.arange(0, 360, 10)
    m.drawparallels(parallels, labels=[True, True, True, False], textcolor='black')
    # ax.text(tx, ty, '{}'.format((group_size[num])))
    meridians = np.arange(0, 360, 20)
    m.drawmeridians(meridians, labels=[True, True, True, True], textcolor='black')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(CS, cax=cbar_ax)
    cbar.set_label('SLP (mbar)')


def plotEOFs(struct):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    # plotting the EOF patterns for SLPs
    plt.figure()
    c1 = 0
    c2 = 0
    for hh in range(9):
        p1 = plt.subplot2grid((3, 3), (c1, c2))
        spatialField = np.multiply(struct.EOFs[hh, 0:len(struct.xFlat[~struct.isOnLandFlat])], np.sqrt(struct.variance[hh]))

        if struct.basin == 'atlantic':
            X_in = struct.xFlat[~struct.isOnLandFlat]
            X_in_Checker = np.where(X_in < 180)
            X_in[X_in_Checker] = X_in[X_in_Checker] + 360

        else:
            X_in = struct.xFlat[~struct.isOnLandFlat]

        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        if struct.basin == 'atlantic':
            m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft, urcrnrlon=struct.lonRight+360, lat_ts=10,
                        resolution='c')
        else:
            m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft, urcrnrlon=struct.lonRight, lat_ts=10,
                        resolution='c')
        if struct.basin == 'atlantic':
            xGridCheck = struct.xGrid
            indexChecker = np.where(xGridCheck<180)
            xGridCheck[indexChecker] = xGridCheck[indexChecker]+360
            cx, cy = m(xGridCheck, struct.yGrid)
        else:
            cx, cy = m(struct.xGrid, struct.yGrid)

        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('SLP EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        c2 += 1
        if c2 == 3:
            c1 += 1
            c2 = 0

    # plotting the EOF patterns for GRDs
    plt.figure()
    c1 = 0
    c2 = 0
    for hh in range(9):
        p1 = plt.subplot2grid((3, 3), (c1, c2))
        spatialField = np.multiply(struct.EOFs[hh, len(struct.xFlat[~struct.isOnLandFlat]):], np.sqrt(struct.variance[hh]))

        if struct.basin == 'atlantic':
            X_in = struct.xFlat[~struct.isOnLandFlat]
            X_in_Checker = np.where(X_in < 180)
            X_in[X_in_Checker] = X_in[X_in_Checker] + 360

        else:
            X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]


        clevels = np.arange(-2, 2, .05)
        if struct.basin == 'atlantic':
            m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft, urcrnrlon=struct.lonRight+360, lat_ts=10,
                        resolution='c')
        else:
            m = Basemap(projection='merc', llcrnrlat=struct.latBot, urcrnrlat=struct.latTop, llcrnrlon=struct.lonLeft, urcrnrlon=struct.lonRight, lat_ts=10,
                        resolution='c')
        if struct.basin == 'atlantic':
            xGridCheck = struct.xGrid
            indexChecker = np.where(xGridCheck < 180)
            xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
            cx, cy = m(xGridCheck, struct.yGrid)
        else:
            cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('GRD EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        c2 += 1
        if c2 == 3:
            c1 += 1
            c2 = 0



def plotEOFsAndPCs(struct):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    # plotting the EOF patterns for SLPs
    plt.figure()
    c1 = 0
    c2 = 0
    for hh in range(9):
        p1 = plt.subplot2grid((9, 3), (c1, c2),rowspan=2,colspan=1)
        spatialField = np.multiply(struct.EOFs[hh, 0:len(struct.xFlat[~struct.isOnLandFlat])], np.sqrt(struct.variance[hh]))

        if struct.basin == 'atlantic':
            X_in = struct.xFlat[~struct.isOnLandFlat]
            X_in_Checker = np.where(X_in < 180)
            X_in[X_in_Checker] = X_in[X_in_Checker] + 360

        else:
            X_in = struct.xFlat[~struct.isOnLandFlat]

        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        # m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
        #             resolution='c')
        m = Basemap(projection='merc', llcrnrlat=5, urcrnrlat=45, llcrnrlon=255, urcrnrlon=360, lat_ts=20,
                    resolution='c')
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        p1 = plt.subplot2grid((9, 3), (c1+2, c2),rowspan=1,colspan=1)
        p1.plot(struct.DATES, struct.PCs[:, hh])
        c2 += 1
        if c2 == 3:
            c1 += 3
            c2 = 0


    #
    # c1 = 3
    # c2 = 0
    # for hh in range(9):
    #     p1 = plt.subplot2grid((9, 3), (c1, c2),rowspan=1,colspan=1)
    #     p1.plot(struct.DATES,struct.PCs[:,hh])
    #     # spatialField = np.multiply(struct.EOFs[hh, 0:len(struct.xFlat[~struct.isOnLandFlat])], np.sqrt(struct.variance[hh]))
    #     # X_in = struct.xFlat[~struct.isOnLandFlat]
    #     # Y_in = struct.yFlat[~struct.isOnLandFlat]
    #     # sea_nodes = []
    #     # for qq in range(len(X_in)):
    #     #     sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))
    #     #
    #     # rectField = np.ones((np.shape(struct.xGrid))) * np.nan
    #     # for tt in range(len(sea_nodes)):
    #     #     rectField[sea_nodes[tt]] = spatialField[tt]
    #     #
    #     # clevels = np.arange(-2, 2, .05)
    #     # m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
    #     #             resolution='c')
    #     # cx, cy = m(struct.xGrid, struct.yGrid)
    #     # m.drawcoastlines()
    #     # CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
    #     # # p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
    #     c2 += 1
    #     if c2 == 3:
    #         c1 += 3
    #         c2 = 0



def plotSeasonal(struct):
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    def GenOneYearDaily(yy=1981, month_ini=1,avgTime=24):
        'returns one generic year in a list of datetimes. Daily resolution'

        dp1 = datetime(yy, month_ini, 2)
        dp2 = dp1 + timedelta(days=364)

        return [dp1 + timedelta(hours=int(avgTime)*i) for i in range(int((dp2 - dp1).days*(24/avgTime)))]


    def GenOneSeasonDaily(yy=1981, month_ini=1):
        'returns one generic year in a list of datetimes. Daily resolution'

        dp1 = datetime(yy, month_ini, 1)
        dp2 = dp1 + timedelta(3 * 365 / 12)

        return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]

    # tC = [datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's')) for qq in struct.DATES]
    tC = struct.DATES

    bmus_dates_months = np.array([d.month for d in tC])
    bmus_dates_days = np.array([d.day for d in tC])


    # generate perpetual year list
    list_pyear = GenOneYearDaily(month_ini=6)
    # m_plot = np.zeros((struct.numClusters, len(list_pyear))) * np.nan
    m_plot = np.zeros((int(np.max(struct.bmus_corrected)+1), len(list_pyear))) * np.nan

    numberOfSims = 1
    # sort data
    for i, dpy in enumerate(list_pyear):
       _, s = np.where(
          [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
       )
       b = struct.bmus_corrected[s]
       b = b.flatten()

       for j in range(int(np.max(struct.bmus_corrected)+1)):
          _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

          m_plot[j, i] = float(len(bb) / float(numberOfSims)) / len(s)

    import matplotlib.cm as cm
    # dwtcolors = cm.viridis(np.linspace(0, 1, struct.numClusters))
    # dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters))
    dwtcolors = cm.rainbow(np.linspace(0, 1, int(np.max(struct.bmus_corrected)+1)))


    fig = plt.figure()
    ax = plt.subplot2grid((1,1),(0,0))
    # plot stacked bars
    bottom_val = np.zeros(m_plot[1, :].shape)
    for r in range(int(np.max(struct.bmus_corrected)+1)):
       row_val = m_plot[r, :]
       ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
       # store bottom
       bottom_val += row_val

    import matplotlib.dates as mdates
    # customize  axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    ax.set_xlim(list_pyear[0], list_pyear[-1])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.set_ylim(0, 1)#struct.endTime[0]-struct.startTime[0])
    ax.set_ylabel('')




def plotSeasonalValidation(struct,numOfSims,avgTime):
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    def GenOneYearDaily(yy=1981, month_ini=1,avgTime=24):
        'returns one generic year in a list of datetimes. Daily resolution'

        dp1 = datetime(yy, month_ini, 2)
        dp2 = dp1 + timedelta(days=364)

        return [dp1 + timedelta(hours=int(avgTime)*i) for i in range(int((dp2 - dp1).days*(24/avgTime)))]

    def GenOneSeasonDaily(yy=1981, month_ini=1):
        'returns one generic year in a list of datetimes. Daily resolution'

        dp1 = datetime(yy, month_ini, 1)
        dp2 = dp1 + timedelta(3 * 365 / 12)

        return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]

    # tC = [datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's')) for qq in struct.DATES]
    # tC = np.asarray(struct.futureDatesSim)
    tC = np.asarray(struct.historicalDatesSim)

    bmus_dates_months = np.array([d.month for d in tC])
    bmus_dates_days = np.array([d.day for d in tC])
    bmus_dates_hours = np.array([d.hour for d in tC])


    # generate perpetual year list
    list_pyear = GenOneYearDaily(month_ini=6,avgTime=avgTime)
    # m_plot = np.zeros((struct.numClusters, len(list_pyear))) * np.nan
    m_plot = np.zeros((int(np.max(struct.bmus_corrected)+1), len(list_pyear))) * np.nan

    numberOfSims = numOfSims
    # compressedBmus = np.hstack(struct.historicalBmusSim)-1
    compressedBmus = struct.historicalBmusSim-1

    # compressedBmus = [np.vstack((compressedBmus,tt.flatten())) for tt in struct.futureBmusSim]
    # sort data
    for i, dpy in enumerate(list_pyear):
       _, s = np.where(
          [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day) & (bmus_dates_hours == dpy.hour)]
       )
       # b = struct.bmus_corrected[s]
       b = compressedBmus[s,:]
       b = b.flatten()

       for j in range(int(np.max(struct.bmus_corrected)+1)):
          _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

          m_plot[j, i] = float(len(bb) / float(numberOfSims)) / len(s)

    import matplotlib.cm as cm
    # dwtcolors = cm.viridis(np.linspace(0, 1, struct.numClusters))
    # dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters))
    dwtcolors = cm.rainbow(np.linspace(0, 1, int(np.max(struct.bmus_corrected)+1)))


    fig = plt.figure()

    ax = plt.subplot2grid((1,1),(0,0))
    # plot stacked bars
    bottom_val = np.zeros(m_plot[1, :].shape)
    for r in range(int(np.max(struct.bmus_corrected)+1)):
       row_val = m_plot[r, :]
       ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
       # store bottom
       bottom_val += row_val

    import matplotlib.dates as mdates
    # customize  axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    ax.set_xlim(list_pyear[0], list_pyear[-1])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    # ax.set_ylim(0, 1)#struct.endTime[0]-struct.startTime[0])
    ax.set_ylabel('')







def plotTotalProbabilityValidation(struct,simNum):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    numClusters = struct.numClusters
    bmus = struct.bmus_corrected  # [1:]
    # evbmus_sim = np.hstack(struct.historicalBmusSim)-1 #evbmus_sim  # evbmus_simALR.T
    evbmus_sim = struct.historicalBmusSim-1 #evbmus_sim  # evbmus_simALR.T

    # Lets make a plot comparing probabilities in sim vs. historical
    probH = np.nan * np.ones((numClusters,))
    probS = np.nan * np.ones((simNum, numClusters))
    for h in np.unique(bmus):
        findH = np.where((bmus == h))[0][:]
        probH[int(h)] = len(findH) / len(bmus)

        for s in range(simNum):
            findS = np.where((evbmus_sim[:,s] == h))[0][:]
            probS[s, int(h)] = len(findS) / len(evbmus_sim)


    # from alrPlotting import colors_mjo
    # from alrPlotting import colors_awt
    dwtcolors = cm.rainbow(np.linspace(0, 1, numClusters))  # 70-20))

    plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    tempPs = np.nan * np.ones((numClusters,))
    for i in range(numClusters):
        temp = probS[:, i]
        temp2 = probH[i]
        box1 = ax.boxplot(temp, positions=[temp2], widths=.002, notch=True, patch_artist=True, showfliers=False)
        plt.setp(box1['boxes'], color=dwtcolors[i])
        plt.setp(box1['means'], color=dwtcolors[i])
        plt.setp(box1['fliers'], color=dwtcolors[i])
        plt.setp(box1['whiskers'], color=dwtcolors[i])
        plt.setp(box1['caps'], color=dwtcolors[i])
        plt.setp(box1['medians'], color=dwtcolors[i], linewidth=0)
        tempPs[i] = np.mean(temp)
        # box1['boxes'].set(facecolor=dwtcolors[i])
        # plt.set(box1['fliers'],markeredgecolor=dwtcolors[i])
    ax.plot([0, 0.045], [0, 0.045], 'k--', zorder=10)
    plt.xlim([0, 0.045])
    plt.ylim([0, 0.045])
    plt.xticks([0, 0.025], ['0', '0.025'])
    plt.xlabel('Historical Probability')
    plt.ylabel('Simulated Probability')
    plt.title('Validation of ALR DWT Simulations')
    return probH, probS
#
#
# def axplot_AWT_2D(ax, var_2D, num_wts, id_wt, color_wt):
#     'axes plot AWT variable (2D)'
#
#     # plot 2D AWT
#     ax.pcolormesh(
#         var_2D,
#         cmap='RdBu_r', shading='gouraud',
#         vmin=-1.5, vmax=+1.5,
#     )
#
#     # title and axis labels/ticks
#     ax.set_title(
#         'WT #{0} --- {1} years'.format(id_wt, num_wts),
#         {'fontsize': 14, 'fontweight':'bold'}
#     )
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_ylabel('month', {'fontsize':8})
#     ax.set_xlabel('lon', {'fontsize':8})
#
#     # set WT color on axis frame
#     plt.setp(ax.spines.values(), color=color_wt, linewidth=4)
#     plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color_wt)
#
# def colors_awt():
#     import numpy as np
#     # 6 AWT colors
#     l_colors_dwt = [
#         (155/255.0, 0, 0),
#         (1, 0, 0),
#         (255/255.0, 216/255.0, 181/255.0),
#         (164/255.0, 226/255.0, 231/255.0),
#         (0/255.0, 190/255.0, 255/255.0),
#         (51/255.0, 0/255.0, 207/255.0),
#     ]
#
#     return np.array(l_colors_dwt)
# def Plot_AWTs(bmus, Km, n_clusters, lon, show=True):
#     '''
#     Plot Annual Weather Types
#
#     bmus, Km, n_clusters, lon - from KMA_simple()
#     '''
#
#     # get number of rows and cols for gridplot
#     #n_cols, n_rows = GetBestRowsCols(n_clusters)
#     n_rows = 2
#     n_cols = 3
#     # get cluster colors
#     cs_awt = colors_awt()
#
#     # plot figure
#     fig = plt.figure()#figsize=(_faspect*_fsize, _fsize))
#
#     gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)
#     gr, gc = 0, 0
#
#     for ic in range(n_clusters):
#
#         id_AWT = ic + 1           # cluster ID
#         index = np.where(bmus==ic)[0][:]
#         var_AWT = Km[ic,:]
#         var_AWT_2D = var_AWT.reshape(-1, len(lon))
#         num_WTs = len(index)
#         clr = cs_awt[ic]          # cluster color
#
#         # AWT var 2D
#         ax = plt.subplot(gs[gr, gc])
#         axplot_AWT_2D(ax, var_AWT_2D, num_WTs, id_AWT, clr)
#
#         gc += 1
#         if gc >= n_cols:
#             gc = 0
#             gr += 1
#
#     # show and return figure
#     if show: plt.show()
#     return fig
# def axplot_AWT_years(ax, dates_wt, bmus_wt, color_wt, xticks_clean=False,
#                      ylab=None, xlims=None):
#     'axes plot AWT dates'
#
#     # date axis locator
#     yloc5 = mdates.YearLocator(5)
#     yloc1 = mdates.YearLocator(1)
#     yfmt = mdates.DateFormatter('%Y')
#
#     # get years string
#     ys_str = np.array([str(d).split('-')[0] for d in dates_wt])
#
#     # use a text bottom - top cycler
#     text_cycler_va = itertools.cycle(['bottom', 'top'])
#     text_cycler_ha = itertools.cycle(['left', 'right'])
#
#     # plot AWT dates and bmus
#     ax.plot(
#         dates_wt, bmus_wt,
#         marker='+',markersize=9, linestyle='', color=color_wt,
#     )
#     va = 'bottom'
#     for tx,ty,tt in zip(dates_wt, bmus_wt, ys_str):
#         ax.text(
#             tx, ty, tt,
#             {'fontsize':8},
#             verticalalignment = next(text_cycler_va),
#             horizontalalignment = next(text_cycler_ha),
#             rotation=45,
#         )
#
#     # configure axis
#     ax.set_yticks([])
#     ax.xaxis.set_major_locator(yloc5)
#     ax.xaxis.set_minor_locator(yloc1)
#     ax.xaxis.set_major_formatter(yfmt)
#     #ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
#     ax.tick_params(axis='x', which='major', labelsize=8)
#
#     # optional parameters
#     if xticks_clean:
#         ax.set_xticklabels([])
#     else:
#         ax.set_xlabel('Year', {'fontsize':8})
#
#     if ylab: ax.set_ylabel(ylab)
#
#     if xlims is not None:
#         ax.set_xlim(xlims[0], xlims[1])
#
# def Plot_AWTs_Dates(bmus, dates, n_clusters, show=True):
#     '''
#     Plot Annual Weather Types dates
#
#     bmus, dates, n_clusters - from KMA_simple()
#     '''
#
#     # get cluster colors
#     cs_awt = colors_awt()
#
#     # plot figure
#     fig, axs = plt.subplots(nrows=n_clusters)#, figsize=(_faspect*_fsize, _fsize))
#
#     # each cluster has a figure
#     for ic in range(n_clusters):
#
#         id_AWT = ic + 1           # cluster ID
#         index = np.where(bmus==ic)[0][:]
#         dates_AWT = dates[index]  # cluster dates
#         bmus_AWT = bmus[index]    # cluster bmus
#         clr = cs_awt[ic]          # cluster color
#
#         ylabel = "WT #{0}".format(id_AWT)
#         #xlims = [dates[0].astype('datetime64[Y]')-np.timedelta64(3, 'Y'), dates[-1].astype('datetime64[Y]')+np.timedelta64(3, 'Y')]
#         xlims = [datetime.datetime(1877,1,1),datetime.datetime(2024,1,1)]
#
#         xaxis_clean=True
#         if ic == n_clusters-1:
#             xaxis_clean=False
#
#         # axs plot
#         axplot_AWT_years(
#             axs[ic], dates_AWT, bmus_AWT,
#             clr, xaxis_clean, ylabel, xlims
#         )
#         #axs[ic].set_xticks(dates_AWT)
#     # show and return figure
#     if show: plt.show()
#     return fig