


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


def plotWTs(struct):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from mpl_toolkits.basemap import Basemap
    import numpy as np

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
        num = struct.kmaOrder[hh]

        spatialField = struct.Km_[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
            struct.xFlat[~struct.isOnLandFlat])] / 100
        # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))
        X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-32, 32, 1)
        m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
                    resolution='c')

        m.fillcontinents(color=dwtcolors[hh])
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-15, vmax=15, cmap=cm.RdBu_r)  # , shading='gouraud')
        # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
        tx, ty = m(320, 10)
        ax.text(tx, ty, '{}'.format(struct.groupSize[num]))

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

    rectField = spatialField.reshape(struct.My, struct.Mx)
    # rectField[~is_on_land] = rectField[~is_on_land]*np.nan
    rectFieldMasked = np.where(~struct.isOnLandGrid, rectField, 0)
    m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10,
                resolution='c')
    m.fillcontinents(color=[0.5, 0.5, 0.5])
    cx, cy = m(struct.xGrid, struct.yGrid)

    m.drawcoastlines()
    # m.bluemarble()
    # CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r, shading='gouraud')
    CS = m.contourf(cx.T, cy.T, rectFieldMasked.T, clevels, vmin=975, vmax=1045, cmap=cm.RdBu_r)  # , shading='gouraud')

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
        X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
                    resolution='c')
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
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
        X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
                        resolution='c')
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        c2 += 1
        if c2 == 3:
            c1 += 1
            c2 = 0