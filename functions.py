import datetime
import random
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
from scipy.spatial import distance_matrix
from datetime import timedelta
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.special import ndtri  # norm inv
from scipy.stats import norm, genpareto, t
from statsmodels.distributions.empirical_distribution import ECDF

from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min


from xml.dom import minidom
from urllib.request import urlopen

def get_elements(url, tag_name, attribute_name):
    """Get elements from an XML file"""
    # usock = urllib2.urlopen(url)
    usock = urlopen(url)
    xmldoc = minidom.parse(usock)
    usock.close()
    tags = xmldoc.getElementsByTagName(tag_name)
    attributes = []
    for tag in tags:
        attribute = tag.getAttribute(attribute_name)
        attributes.append(attribute)
    return attributes

def loadWaterLevel(file):
    from netCDF4 import Dataset
    wldata = Dataset(file)
    waterLevel = wldata.variables['waterLevel'][:]
    predictedWaterLevel = wldata.variables['predictedWaterLevel'][:]
    residualWaterLevel = wldata.variables['residualWaterLevel'][:]
    timeWl = wldata.variables['time'][:]
    output = dict()
    output['waterLevel'] = waterLevel
    output['predictedWaterLevel'] = predictedWaterLevel
    output['residualWaterLevel'] = residualWaterLevel
    output['time'] = timeWl
    return output


def loadWIS(file):
    from netCDF4 import Dataset

    waves = Dataset(file)
    waveHs = waves.variables['waveHs'][:]
    waveTp = waves.variables['waveTp'][:]
    waveMeanDirection = waves.variables['waveMeanDirection'][:]
    waveTm = waves.variables['waveTm'][:]
    waveTm1 = waves.variables['waveTm1'][:]
    waveTm2 = waves.variables['waveTm2'][:]
    waveHsWindsea = waves.variables['waveHsWindsea'][:]
    waveTmWindsea = waves.variables['waveTmWindsea'][:]
    waveMeanDirectionWindsea = waves.variables['waveMeanDirectionWindsea'][:]
    waveSpreadWindsea = waves.variables['waveSpreadWindsea'][:]
    timeW = waves.variables['time'][:]
    waveTpSwell = waves.variables['waveTpSwell'][:]
    waveHsSwell = waves.variables['waveHsSwell'][:]
    waveMeanDirectionSwell = waves.variables['waveMeanDirectionSwell'][:]
    waveSpreadSwell = waves.variables['waveSpreadSwell'][:]
    waveSpread = waves.variables['waveSpread'][:]

    windSpeed = waves.variables['windSpeed'][:]
    windDirection = waves.variables['windDirection'][:]

    output = dict()
    output['waveHs'] = waveHs
    output['waveTp'] = waveTp
    output['waveMeanDirection'] = waveMeanDirection
    output['waveTm'] = waveTm
    output['waveTm1'] = waveTm1
    output['waveTm2'] = waveTm2
    output['waveTpSwell'] = waveTpSwell
    output['waveHsSwell'] = waveHsSwell
    output['waveMeanDirectionSwell'] = waveMeanDirectionSwell
    output['waveSpreadSwell'] = waveSpreadSwell
    output['waveHsWindsea'] = waveHsWindsea
    output['waveTpWindsea'] = waveTmWindsea
    output['waveMeanDirectionWindsea'] = waveMeanDirectionWindsea
    output['waveSpreadWindsea'] = waveSpreadWindsea
    output['windSpeed'] = windSpeed
    output['windDirection'] = windDirection
    output['waveSpread'] = waveSpread

    output['t'] = timeW

    return output



def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def dateDay2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetimeDate(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.date(d[0], d[1], d[2]) for d in d_vec]


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out



def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''
    # TODO: DOCUMENTAR.

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                qx += dd[sc[i-1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i-1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i-1,j+1], sc[i,j]]

            # row F
            if not j==0:
                qx += dd[sc[i,j-1], sc[i,j]]

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

            # row F+1
            if not i+1==dimy:
                qx += dd[sc[i+1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i+1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i+1,j+1], sc[i,j]]

    # test permutations
    q=np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i,j,k]))==3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f=0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix==0:
                                f += dd[u[ix-1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix-1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix-1,jx+1], u[ix,jx]]

                            # row F
                            if not jx==0:
                                f += dd[u[ix,jx-1], u[ix,jx]]

                            if not jx+1==dimx:
                                f += dd[u[ix,jx+1], u[ix,jx]]

                            # row F+1
                            if not ix+1==dimy:
                                f += dd[u[ix+1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix+1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix+1,jx+1], u[ix,jx]]

                    if f<=q:
                        q = f
                        sc = u

                        if q<=qx:
                            qx=q
                            go_out=False

    return sc.flatten('F')



def fitGEVparams(var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus        - KMA bmus (time series of KMA centroids)
    n_clusters  - number of KMA clusters
    var         - time series of variable to fit to GEV/Gumbel_L

    returns np.array (n_clusters x parameters). parameters = (shape, loc, scale)
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((3,))

    # get variable at cluster position
    var_c = var
    var_c = var_c[~np.isnan(var_c)]

    # fit to Gumbel_l and get negative loglikelihood
    loc_gl, scale_gl = gumbel_l.fit(-var_c)
    theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
    nLogL_gl = genextreme.nnlf(theta_gl, var_c)

    # fit to GEV and get negative loglikelihood
    c = -0.1
    shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
    theta_gev = (shape_gev, loc_gev, scale_gev)
    nLogL_gev = genextreme.nnlf(theta_gev, var_c)

    # store negative shape
    theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

    # apply significance test if Frechet
    if shape_gev < 0:

        # TODO: cant replicate ML exact solution
        if nLogL_gl - nLogL_gev >= 1.92:
            param_GEV = list(theta_gev_fix)
        else:
            param_GEV = list(theta_gl)
    else:
        param_GEV = list(theta_gev_fix)

    return param_GEV


def gev_CDF(x):
    '''
    :param x: observations
    :return: normalized cdf
    '''
    shape, loc, scale = fitGEVparams(x)
    cdf = genextreme.cdf(x, -1 * shape, loc, scale)
    return cdf

def gev_ICDF(x,y):
    '''
    :param x: observations
    :param y: simulated probabilities
    :return: simulated values
    '''
    shape, loc, scale = fitGEVparams(x)
    ppf_VV = genextreme.ppf(y, -1 * shape, loc, scale)
    return ppf_VV


def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF at x position (kde.support = x)
    fint = interp1d(kde.support, kde.cdf)
    return fint(x)

def ksdensity_ICDF(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)
    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]
    return fint(p)

def GeneralizedPareto_CDF(x):
    '''
    Generalized Pareto fit
    Returns cumulative probability function at x.
    '''

    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)

    # get generalized pareto CDF
    cdf = genpareto.cdf(x, shape, scale=scale)

    return cdf

def GeneralizedPareto_ICDF(x, p):
    '''
    Generalized Pareto fit
    Returns inverse cumulative probability function at p points
    '''
    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)
    # get percent points (inverse of CDF)
    icdf = genpareto.ppf(p, shape, scale=scale)
    return icdf

def Empirical_CDF(x):
    '''
    Returns empirical cumulative probability function at x.
    '''
    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)
    return cdf

def Empirical_ICDF(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: build in functionality for a fill_value?

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)


def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student)

    if family=='gaussian':
        u[u>=1.0] = 0.999999
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # # :
        # x = np.linspace(np.min(u), np.max(u),100)
        # inv_t = np.ndarray((len(x), u.shape[1]))
        #
        # for j in range(u.shape[1]):
        #     param = t.fit(u[:,j])
        #     t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
        #     inv_t[:,j] = t_pdf
        #
        # # CORRELATION? NUHAT?
        # rhohat = np.corrcoef(inv_n.T)
        # nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat

def copularnd(family, rhohat, n):
    '''
    Random vectors from a copula
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u


def copulaSimulation(U_data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    U_data: 2D nump.array, each variable in a column
    kernels: list of kernels for each column at U_data (KDE | GPareto | Empirical | GEV)
    num_sim: number of simulations
    '''

    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_CDF, ksdensity_ICDF),
        'GPareto' : (GeneralizedPareto_CDF, GeneralizedPareto_ICDF),
        'ECDF' : (Empirical_CDF, Empirical_ICDF),
        'GEV': (gev_CDF, gev_ICDF),
    }


    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernels, ' | '.join(d_kf.keys())
            )
        )


    # normalize: calculate data CDF using kernels
    U_cdf = np.zeros(U_data.shape) * np.nan
    ic = 0
    for d, k in zip(U_data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula
    rhohat, _ = copulafit(U_cdf, 'gaussian')

    # simulate data to fill probabilistic space
    U_cop = copularnd('gaussian', rhohat, num_sim)

    # de-normalize: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(U_data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim





def xds2datetime(d64):
    from datetime import datetime
    'converts xr.Dataset np.datetime64[ns] into datetime'
    # TODO: hour minutes and seconds

    return datetime(int(d64.dt.year), int(d64.dt.month), int(d64.dt.day))


def xds_reindex_daily(xds_data,  dt_lim1=None, dt_lim2=None):
    from datetime import datetime
    '''
    Reindex xarray.Dataset to daily data between optional limits
    '''

    # TODO: remove limits from inside function

    # TODO: remove this swich and use date2datenum
    if isinstance(xds_data.time.values[0], datetime):
        xds_dt1 = xds_data.time.values[0]
        xds_dt2 = xds_data.time.values[-1]
    else:
        # parse xds times to python datetime
        xds_dt1 = xds2datetime(xds_data.time[0])
        xds_dt2 = xds2datetime(xds_data.time[-1])

    # cut data at limits
    if dt_lim1:
        xds_dt1 = max(xds_dt1, dt_lim1)
    if dt_lim2:
        xds_dt2 = min(xds_dt2, dt_lim2)

    # number of days
    num_days = (xds_dt2-xds_dt1).days+1

    # reindex xarray.Dataset
    return xds_data.reindex(
        {'time': [xds_dt1 + timedelta(days=i) for i in range(num_days)]},
        method = 'pad',
    )



def xds_reindex_flexible(xds_data,  dt_lim1=None, dt_lim2=None,avgTime=None):
    from datetime import datetime
    '''
    Reindex xarray.Dataset to daily data between optional limits
    '''

    # TODO: remove limits from inside function

    # TODO: remove this swich and use date2datenum
    if isinstance(xds_data.time.values[0], datetime):
        xds_dt1 = xds_data.time.values[0]
        xds_dt2 = xds_data.time.values[-1]
    else:
        # parse xds times to python datetime
        xds_dt1 = xds2datetime(xds_data.time[0])
        xds_dt2 = xds2datetime(xds_data.time[-1])

    # cut data at limits
    if dt_lim1:
        xds_dt1 = max(xds_dt1, dt_lim1)
    if dt_lim2:
        xds_dt2 = min(xds_dt2, dt_lim2)

    # number of days
    # num_days = (xds_dt2-xds_dt1).days+1
    num_windows = int(((xds_dt2-xds_dt1).days)*(24/avgTime)+(24/avgTime))
    # reindex xarray.Dataset
    return xds_data.reindex(
        {'time': [xds_dt1 + timedelta(hours=int(avgTime*i)) for i in range(num_windows)]},
        method = 'pad',
    )


def xds_common_dates_daily(xds_list):
    from datetime import timedelta
    '''
    returns daily datetime array between a list of xarray.Dataset comon date
    limits
    '''
    d1, d2 = xds_limit_dates(xds_list)
    return [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]


def xds_common_dates_flexible(xds_list,avgTime=None):
    from datetime import timedelta
    '''
    returns daily datetime array between a list of xarray.Dataset comon date
    limits
    '''
    d1, d2 = xds_limit_dates(xds_list)
    print('d1 = {}'.format(d1))
    print('d2 = {}'.format(d2))
    return [d1 + timedelta(hours=int(avgTime*i)) for i in range(int((d2-d1).days*(24/avgTime)+(24/avgTime)))]

def xds_limit_dates(xds_list):
    '''
    returns datetime common limits between a list of xarray.Dataset
    '''
    from datetime import datetime
    d1 = None
    d2 = None

    for xds_e in xds_list:

        # TODO: remove this swich and use date2datenum
        if isinstance(xds_e.time.values[0], datetime):
            xds_e_dt1 = xds_e.time.values[0]
            xds_e_dt2 = xds_e.time.values[-1]
        else:
            # parse xds times to python datetime
            xds_e_dt1 = xds2datetime(xds_e.time[0])
            xds_e_dt2 = xds2datetime(xds_e.time[-1])

        if d1 == None:
            d1 = xds_e_dt1
            d2 = xds_e_dt2

        d1 = max(xds_e_dt1, d1)
        d2 = min(xds_e_dt2, d2)

    return d1, d2



def npdt64todatetime(dt64):
    from datetime import datetime
    from datetime import timedelta
    'converts np.datetime64[ns] into datetime'

    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime(1970, 1, 1) + timedelta(seconds=ts)




def return_value(sample_real, threshold, alpha, block_size, return_period,
                 fit_method):  # return value plot and return value estimative
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector
    import math as mt
    from scipy.stats import norm

    POT = importr('POT')  # importing POT package
    sample = np.sort(sample_real)
    sample_excess = []
    sample_over_thresh = []
    for data in sample:
      if data > threshold + 0.00001:
         sample_excess.append(data - threshold)
         sample_over_thresh.append(data)

    rdata = FloatVector(sample)
    fit = POT.fitgpd(rdata, threshold, est=fit_method)  # fit data
    shape = fit[0][1]
    scale = fit[0][0]

    # Computing the return value for a given return period with the confidence interval estimated by the Delta Method
    m = return_period
    Eu = len(sample_over_thresh) / len(sample)
    x_m = threshold + (scale / shape) * (((m * Eu) ** shape) - 1)

    # Solving the delta method
    d = Eu * (1 - Eu) / len(sample)
    e = fit[3][0]
    f = fit[3][1]
    g = fit[3][2]
    h = fit[3][3]
    a = (scale * (m ** shape)) * (Eu ** (shape - 1))
    b = (shape ** -1) * (((m * Eu) ** shape) - 1)
    c = (-scale * (shape ** -2)) * ((m * Eu) ** shape - 1) + (scale * (shape ** -1)) * ((m * Eu) ** shape) * mt.log(
      m * Eu)
    CI = (norm.ppf(1 - (alpha / 2)) * ((((a ** 2) * d) + (b * ((c * g) + (e * b))) + (c * ((b * f) + (c * h)))) ** 0.5))

    print('The return value for the given return period is {} \u00B1 {}'.format(x_m, CI))

    ny = block_size  # defining how much observations will be a block (usually anual)
    N_year = return_period / block_size  # N_year represents the number of years based on the given return_period

    for i in range(0, len(sample)):
      if sample[i] > threshold + 0.0001:
         i_initial = i
         break

    p = np.arange(i_initial, len(sample)) / (len(sample))  # Getting Plotting Position points
    N = 1 / (ny * (1 - p))  # transforming plotting position points to years

    year_array = np.arange(min(N), N_year + 0.1, 0.1)  # defining a year array

    # Algorithm to compute the return value and the confidence intervals for plotting
    z_N = []
    CI_z_N_high_year = []
    CI_z_N_low_year = []
    for year in year_array:
      z_N.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1))
      a = (scale * ((year * ny) ** shape)) * (Eu ** (shape - 1))
      b = (shape ** -1) * ((((year * ny) * Eu) ** shape) - 1)
      c = (-scale * (shape ** -2)) * (((year * ny) * Eu) ** shape - 1) + (scale * (shape ** -1)) * (
                 ((year * ny) * Eu) ** shape) * mt.log((year * ny) * Eu)
      CIyear = (norm.ppf(1 - (alpha / 2)) * (
                 (((a ** 2) * d) + (b * ((c * g) + (e * b))) + (c * ((b * f) + (c * h)))) ** 0.5))
      CI_z_N_high_year.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1) + CIyear)
      CI_z_N_low_year.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1) - CIyear)

    # Plotting Return Value
    # plt.figure(8)
    # plt.plot(year_array, CI_z_N_high_year, linestyle='--', color='red', alpha=0.8, lw=0.9, label='Confidence Bands')
    # plt.plot(year_array, CI_z_N_low_year, linestyle='--', color='red', alpha=0.8, lw=0.9)
    # plt.plot(year_array, z_N, color='black', label='Theoretical Return Level')
    # plt.scatter(N, sample_over_thresh, label='Empirical Return Level')
    # plt.xscale('log')
    # plt.xlabel('Return Period')
    # plt.ylabel('Return Level')
    # plt.title('Return Level Plot')
    # plt.legend()
    #
    # plt.show()

    output = dict()
    output['year_array'] = year_array
    output['N'] = N
    output['sample_over_thresh'] = sample_over_thresh
    output['CI_z_N_high_year'] = CI_z_N_high_year
    output['CI_z_N_low_year'] = CI_z_N_low_year
    output['z_N'] = z_N
    output['CI'] = CI
    return output














# TODO REFACTOR CON teslakit/database.py

def StoreBugXdset(xds_data, p_ncfile):
    # common
    from datetime import datetime, date
    import os
    import os.path as op

    # pip
    import netCDF4
    import numpy as np

    # tk
    #from functions import npdt64todatetime as n2d

    '''
    Stores xarray.Dataset to .nc file while avoiding bug with time data (>2262)
    '''

    # get metadata from xarray.Dataset
    dim_names = xds_data.dims.keys()
    var_names = xds_data.variables.keys()

    # Handle time data  (calendar format)
    calendar = 'standard'
    units = 'hours since 1970-01-01 00:00:00'

    # remove previous file
    if op.isfile(p_ncfile):
        os.remove(p_ncfile)

    # Use netCDF4 lib
    root = netCDF4.Dataset(p_ncfile, 'w', format='NETCDF4')

    # Handle dimensions
    for dn in dim_names:
        vals = xds_data[dn].values[:]
        root.createDimension(dn, len(vals))

    # handle variables
    for vn in var_names:
        vals = xds_data[vn].values[:]

        # dimensions values
        if vn in dim_names:

            if vn == 'time':  # time dimension values
                # TODO: profile / acelerar
                if isinstance(vals[0], datetime):
                    pass

                elif isinstance(vals[0], date):
                    # TODO se pierde la resolucion horaria
                    # parse datetime.date to datetime.datetime
                    vals = [datetime.combine(d, datetime.min.time()) for d in vals]

                elif isinstance(vals[0], np.datetime64):
                    # parse numpy.datetime64 to datetime.datetime
                    vals = [npdt64todatetime(d) for d in vals]

                dv = root.createVariable(varname=vn, dimensions=(vn,), datatype='int64')
                dv[:] = netCDF4.date2num(vals, units=units, calendar=calendar)
                dv.units = units
                dv.calendar = calendar

            else:
                dv = root.createVariable(varname=vn, dimensions=(vn,), datatype=type(vals[0]))
                dv[:] = vals

        # variables values
        else:
            vdims = xds_data[vn].dims
            vatts = xds_data[vn].attrs

            vv = root.createVariable(varname=vn,dimensions=vdims, datatype='float32')
            vv[:] = vals

            # variable attributes
            vv.setncatts(vatts)

    # global attributes
    root.setncatts(xds_data.attrs)

    # close file
    root.close()




def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out


# common
import pickle
import time
import sys
import os
import os.path as op
from collections import OrderedDict
from datetime import datetime, date, timedelta

# pip
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.discrete.discrete_model as smDD
import scipy.stats as stat
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator)
import matplotlib.colors as mcolors
from matplotlib import cm

# fix tqdm for notebook
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

# fix library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# tk

# from alrPlotting import Plot_PValues, Plot_Params, Plot_Terms, Plot_Compare_Covariate, Plot_Log_Sim
# from wtsPlotting import Plot_Compare_PerpYear, Plot_Compare_Transitions, Plot_Compare_Persistences


_faspect = 1.618
_fsize = 9.8
_fdpi = 128

def colors_awt():

    # 6 AWT colors
    l_colors_dwt = [
        (155/255.0, 0, 0),
        (1, 0, 0),
        (255/255.0, 216/255.0, 181/255.0),
        (164/255.0, 226/255.0, 231/255.0),
        (0/255.0, 190/255.0, 255/255.0),
        (51/255.0, 0/255.0, 207/255.0),
    ]

    return np.array(l_colors_dwt)

def colors_mjo():
    'custom colors for MJO 25 categories'

    l_named_colors = [
        'lightskyblue', 'deepskyblue', 'royalblue', 'mediumblue',
        'darkblue', 'darkblue', 'darkturquoise', 'turquoise',
        'maroon', 'saddlebrown', 'chocolate', 'gold', 'orange',
        'orangered', 'red', 'firebrick', 'Purple', 'darkorchid',
        'mediumorchid', 'magenta', 'mediumslateblue', 'blueviolet',
        'darkslateblue', 'indigo', 'darkgray',
    ]

    # get rgb colors as numpy array
    np_colors_rgb = np.array(
        [mcolors.to_rgb(c) for c in l_named_colors]
    )

    return np_colors_rgb

def colors_interp(num_clusters):

    # generate spectral colormap
    scm = cm.get_cmap('Spectral', num_clusters)

    # use normalize
    mnorm = mcolors.Normalize(vmin=0, vmax=num_clusters)

    # interpolate colors from cmap
    l_colors = []
    for i in range(num_clusters):
        l_colors.append(scm(mnorm(i)))

    # return numpy array
    np_colors_rgb = np.array(l_colors)[:,:-1]

    return np_colors_rgb


def GetClusterColors(num_clusters):
    'Choose colors or Interpolate custom colormap to number of clusters'

    if num_clusters == 6:
        np_colors_rgb = colors_awt()  # Annual Weather Types

    if num_clusters == 25:
        np_colors_rgb = colors_mjo()  # MJO Categories

    elif num_clusters in [36, 42, 48]:
        etcolors = cm.viridis(np.linspace(0, 1, 48 - 11))
        tccolors = np.flipud(cm.autumn(np.linspace(0, 1, 12)))

        np_colors_rgb = np.vstack((etcolors, tccolors[1:, :]))

        #np_colors_rgb = colors_dwt(num_clusters)  # Daily Weather Types

    else:
        np_colors_rgb = colors_interp(num_clusters)  # interpolate

    return np_colors_rgb


class MidpointNormalize(mcolors.Normalize):
	'''
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	'''

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		mcolors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def Generate_Covariate_Matrix(
    bmus_values, covar_values,
    bmus_dates, covar_dates,
    num_clusters, covar_rng, num_sim=1):
    'Calculates and returns matrix for stacked bar plotting'

    # generate aux arrays
    #bmus_years = [d.year for d in bmus_dates]
    m_plot = np.zeros((num_clusters, len(covar_rng)-1)) * np.nan

    for i in range(len(covar_rng)-1):

        # find years inside range
        _, s = np.where(
            [(covar_values >= covar_rng[i]) & (covar_values <= covar_rng[i+1])]
        )

        # TODO: usando alr_wrapper las fechas covar y bmus coinciden
        b = bmus_values[s,:]
        b = b.flatten()

        # TODO: mejorar, no usar los years y posicion.
        # usar la fecha

        #ys = [covar_dates[x].year for x in s]
        # find data inside years found
        #sb = np.where(np.in1d(bmus_years, ys))[0]
        #b = bmus_values[sb,:]
        #b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])
            # TODO sb se utiliza para el test de laura
            #if len(sb) > 0:
                #m_plot[j,i] = float(len(bb)/float(num_sim))/len(sb)
            if len(s) > 0:
                m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)
            else:
                m_plot[j,i] = 0

    return m_plot

def Generate_Covariate_rng(covar_name, cov_values):
    'Returns covar_rng and interval depending on covar_name'

    if covar_name.startswith('PC'):
        delta = 5
        n_rng = 7

        covar_rng = np.linspace(
            np.min(cov_values)-delta,
            np.max(cov_values)+delta,
            n_rng
        )

    elif covar_name.startswith('MJO'):
        delta = 0.5
        n_rng = 7

        covar_rng = np.linspace(
            np.min(cov_values)-delta,
            np.max(cov_values)+delta,
            n_rng
        )

    else:
        print('Cant plot {0}, missing rng params in plotting library'.format(
            name_covar
        ))
        return None, None

    # interval
    interval = covar_rng[1]-covar_rng[0]

    return covar_rng, interval


def Plot_PValues(p_values, term_names, show=True):
    'Plot ARL/BMUS p-values'

    n_wts = p_values.shape[0]
    n_terms = p_values.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    c = ax.pcolor(p_values, cmap='coolwarm_r', clim=(0,1),
                  norm=MidpointNormalize(midpoint=0.1, vmin=0, vmax=1))
    #c.cmap.set_over('w')
    fig.colorbar(c, ax=ax)

    # Pval text
    #for i in range(p_values.shape[1]):
    #    for j in range(p_values.shape[0]):
    #        v = p_values[j,i]
    #        if v<=0.1:
    #            ax.text(i+0.5, j+0.5, '{0:.2f}'.format(v),
    #                    va='center', ha='center', size=6)

    # axis
    ax.set_title('p-value', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms)+0.5, minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90, fontsize=7)

    ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts)+0.5, minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # add grid
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Params(params, term_names, show=True):
    'Plot ARL/BMUS params'

    n_wts = params.shape[0]
    n_terms = params.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    # text table and color
    #c = ax.pcolor(params, cmap=plt.cm.bwr)
    c = ax.pcolor(
        params, cmap='coolwarm_r',
        norm=MidpointNormalize(midpoint=0)
    )
    #for i in range(params.shape[1]):
    #    for j in range(params.shape[0]):
    #        v = params[j,i]
    #        ax.text(i+0.5, j+0.5, '{0:.1f}'.format(v),
    #                va='center', ha='center', size=6)
    fig.colorbar(c, ax=ax)

    # axis
    ax.set_title('params', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms)+0.5, minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90, fontsize=7)

    ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts)+0.5, minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # add grid
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')

    # show and return figure
    if show: plt.show()
    return fig


def Plot_Log_Sim(log, show=True):
    '''
    Plot ALR simulation log

    log - xarray.Dataset from alr wrapper (n_sim already selected)
    '''

    # plot figure
    #fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    fig = plt.figure(figsize=[18.5,9])

    # figure gridspec
    gs1 = gridspec.GridSpec(4,1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1], sharex=ax1)
    ax3 = fig.add_subplot(gs1[2], sharex=ax1)
    ax4 = fig.add_subplot(gs1[3], sharex=ax1)

    # Plot evbmus values
    ax1.plot(
        log.time, log.evbmus_sims, ':',
        linewidth=0.5, color='grey',
        marker='.', markersize=3,
        markerfacecolor='crimson', markeredgecolor='crimson'
    )

    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.grid(which='major', linestyle=':', alpha=0.5)
    ax1.set_xlim(log.time[0], log.time[-1])
    ax1.set_ylabel('Bmus', fontsize=12)

    # Plot evbmus probabilities
    z = np.diff(np.column_stack(
        ([np.zeros([len(log.time),1]), log.probTrans.values])
    ), axis=1)
    z1 = np.column_stack((z, z[:,-1])).T
    z2 = np.column_stack((z1, z1[:,-1]))
    p1 = ax2.pcolor(
        np.append(log.time, log.time[-1]),
        np.append(log.n_clusters, log.n_clusters[-1]), z2,
        cmap='PuRd', edgecolors='grey', linewidth=0.05
    )
    ax2.set_ylabel('Bmus',fontsize=12)

    # TODO: gestionar terminos markov
    # TODO: no tengo claro si el primero oel ultimo
    alrt0 = log.alr_terms.isel(mk=0)

    # Plot Terms
    for v in range(len(log.terms)):
        if log.terms.values[v].startswith('ss'):
            ax3.plot(log.time, alrt0[:,v], label=log.terms.values[v])

        if log.terms.values[v].startswith('PC'):
            ax4.plot(log.time, alrt0[:,v], label=log.terms.values[v])

        if log.terms.values[v].startswith('MJO'):
            ax4.plot(log.time, alrt0[:,v], label=log.terms.values[v])

    # TODO: plot terms markov??

    ax3.set_ylim(-1.8,1.2)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(loc='lower left',ncol=len(handles))
    ax3.set_ylabel('Seasonality',fontsize=12)

    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(loc='lower left',ncol=len(handles))
    ax4.set_ylabel('Covariates',fontsize=12)
    # cbar=plt.colorbar(p1,ax=ax2,pad=0)
    # cbar.set_label('Transition probability')

    gs1.tight_layout(fig, rect=[0.05, [], 0.95, []])

    # custom colorbar for probability
    gs2 = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs2[0])
    plt.colorbar(p1, cax=ax1)
    ax1.set_ylabel('Probability')
    gs2.tight_layout(fig, rect=[0.935, 0.52, 0.99, 0.73])

    # show and return figure
    if show: plt.show()
    return fig


# TODO: following functions are not finished / tested

def Plot_Covariate(bmus_values, covar_values,
                   bmus_dates, covar_dates,
                   num_clusters, name_covar,
                   num_sims=1,
                   p_export=None):
    'Plots ARL covariate related to bmus stacked bar chart'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate common covar_rng
    covar_rng, interval = Generate_Covariate_rng(
        name_covar, covar_values)

    # generate plot matrix
    m_plot = Generate_Covariate_Matrix(
        bmus_values, covar_values,
        bmus_dates, covar_dates,
        num_clusters, covar_rng, num_sims)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))
    x_val = covar_rng[:-1]

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # axis
    plt.xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    plt.ylim(0, 1)
    plt.xlabel(name_covar)
    plt.ylabel('')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Terms(terms_matrix, terms_dates, terms_names, show=True):
    'Plot terms used for ALR fitting'

    # number of terms
    n_sps = terms_matrix.shape[1]

    # custom fig size
    fsy = n_sps * 2

    # plot figure
    fig, ax_list = plt.subplots(
        n_sps, 1, sharex=True, figsize=(_faspect*_fsize, fsy)
    )

    x = terms_dates
    for i in range(n_sps):
        y = terms_matrix[:,i]
        n = terms_names[i]
        ax = ax_list[i]
        ax.plot(x, y, '.b')
        ax.set_title(n, loc='left', fontweight='bold', fontsize=10)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-1, 1)
        ax.grid(True, which='both')

        if n=='intercept':
            ax.set_ylim(0, 2)

    # date label
    fig.text(0.5, 0.04, 'date (y)', ha='center', fontweight='bold')
    fig.text(0.04, 0.5, 'value (-)', va='center', rotation='vertical',
             fontweight='bold')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Compare_Covariate(num_clusters,
                           bmus_values_sim, bmus_dates_sim,
                           bmus_values_hist, bmus_dates_hist,
                           cov_values_sim, cov_dates_sim,
                           cov_values_hist, cov_dates_hist,
                           name_covar,
                           n_sim = 1, p_export=None):
    'Plot simulated - historical bmus comparison related to covariate'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate common covar_rng
    covar_rng, interval = Generate_Covariate_rng(
        name_covar, np.concatenate((cov_values_sim, cov_values_hist)))

    # generate plot matrix
    m_plot_sim = Generate_Covariate_Matrix(
        bmus_values_sim, cov_values_sim,
        bmus_dates_sim, cov_dates_sim,
        num_clusters, covar_rng, n_sim)

    m_plot_hist = Generate_Covariate_Matrix(
        bmus_values_hist, cov_values_hist,
        bmus_dates_hist, cov_dates_hist,
        num_clusters, covar_rng, 1)

    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(_faspect*_fsize, _fsize))
    x_val = covar_rng[:-1]

    # sim
    bottom_val = np.zeros(m_plot_sim[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_sim[r,:]
        ax_sim.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # hist
    bottom_val = np.zeros(m_plot_hist[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_hist[r,:]
        ax_hist.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np_colors_int[r]
               )

        # store bottom
        bottom_val += row_val

    # axis
    # MEJORAR Y METER EL NOMBRE DE LA COVARIATE
    ax_sim.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_title('Simulation')
    ax_sim.set_ylabel('')
    ax_sim.set_xlabel('')

    ax_hist.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_title('Historical')
    ax_hist.set_ylabel('')
    ax_sim.set_xlabel('')

    fig.suptitle(name_covar, fontweight='bold', fontsize=12)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Persistences(series):
    'Return series persistences for each element'

    # locate dates where series changes
    s_diff = np.diff(series)
    ix_ch = np.where((s_diff != 0))[0]+1
    ix_ch = np.insert(ix_ch, 0, 0)

    wt_ch = series[ix_ch][:-1] # bmus where WT changes
    wt_dr = np.diff(ix_ch)

    # output dict
    d_pers = {}
    for e in set(series):
        d_pers[e] = wt_dr[wt_ch==e]

    return d_pers



class ALR_WRP(object):
    'AutoRegressive Logistic Model Wrapper'

    def __init__(self, p_base):

        # data needed for ALR fitting
        self.xds_bmus_fit = None
        self.cluster_size = None

        # ALR terms
        self.d_terms_settings = {}
        self.terms_fit = {}
        self.terms_fit_names = []

        # temporal data storage
        self.mk_order = 0
        self.cov_names = []

        # ALR model core
        self.model = None

        # config (only tested with statsmodels library)
        self.model_library = 'statsmodels'  # sklearn / statsmodels

        # paths
        self.p_base = p_base

        # alr model and terms_fit
        self.p_save_model = op.join(p_base, 'model.sav')
        self.p_save_terms_fit = op.join(p_base, 'terms.sav')

        # store fit and simulated bmus
        self.p_save_fit_xds = op.join(p_base, 'xds_input.nc')
        self.p_save_sim_xds = op.join(p_base, 'xds_output.nc')

        # export folders for figures
        self.p_report_fit = op.join(p_base, 'report_fit')
        self.p_report_sim = op.join(p_base, 'report_sim')

        # log sim
        self.p_log_sim_xds = op.join(p_base, 'xds_log_sim.nc')

    def SetFitData(self, cluster_size, xds_bmus_fit, d_terms_settings,avgTime):
        '''
        Sets data needed for ALR fitting

        cluster_size - number of clusters in classification
        xds_bmus_fit - xarray.Dataset vars: bmus, dims: time
        d_terms_settings - terms settings. See "SetFittingTerms"
        '''

        self.cluster_size = cluster_size
        self.xds_bmus_fit = xds_bmus_fit
        self.SetFittingTerms(d_terms_settings,avgTime)

        # save bmus series used for fitting
        self.SaveBmus_Fit()

    def SetFittingTerms(self, d_terms_settings,avgTime):
        'Set terms settings that will be used for fitting'

        # default settings used for ALR terms
        default_settings = {
            'mk_order'  : 0,
            'constant' : False,
            'long_term' : False,
            'seasonality': (False, []),
            'covariates': (False, []),
            'covariates_seasonality': (False, []),
        }

        # join user and default input
        for k in default_settings.keys():
            if k not in d_terms_settings:
                d_terms_settings[k] = default_settings[k]

        # generate ALR terms
        bmus_fit = self.xds_bmus_fit.bmus.values
        time_fit = self.xds_bmus_fit.time.values
        cluster_size = self.cluster_size

        self.terms_fit, self.terms_fit_names = self.GenerateALRTerms(
            d_terms_settings, bmus_fit, time_fit, cluster_size, avgTime, time2yfrac=True)

        # store data
        self.mk_order = d_terms_settings['mk_order']
        self.d_terms_settings = d_terms_settings

    def GenerateALRTerms(self, d_terms_settings, bmus, time, cluster_size,avgTime,
                         time2yfrac=False):
        'Generate ALR terms from user terms settings'

        # terms stored at OrderedDict
        terms = OrderedDict()
        terms_names = []

        # time options (time has to bee yearly fraction)
        if time2yfrac:
            time_yfrac = self.GetFracYears(time,avgTime)
        else:
            time_yfrac = time

        # constant term
        if d_terms_settings['constant']:
            terms['constant'] = np.ones((bmus.size, 1))
            terms_names.append('intercept')

        # time term (use custom time array with year decimals)
        if d_terms_settings['long_term']:
            terms['long_term'] = np.ones((bmus.size, 1))
            terms['long_term'][:,0] = time_yfrac
            terms_names.append('long_term')

        # seasonality term
        if d_terms_settings['seasonality'][0]:
            phases  = d_terms_settings['seasonality'][1]
            temp_seas = np.zeros((len(time_yfrac), 2*len(phases)))
            c = 0
            for a in phases:
                temp_seas [:,c]   = np.cos(a * np.pi * time_yfrac)
                temp_seas [:,c+1] = np.sin(a * np.pi * time_yfrac)
                terms_names.append('ss_cos_{0}'.format(a))
                terms_names.append('ss_sin_{0}'.format(a))
                c+=2
            terms['seasonality'] = temp_seas

        # Covariates term
        if d_terms_settings['covariates'][0]:

            # covariates dataset (vars: cov_values, dims: time, cov_names)
            xds_cov = d_terms_settings['covariates'][1]
            cov_names = xds_cov.cov_names.values
            self.cov_names = cov_names  # storage

            # normalize covars
            if not 'cov_norm' in xds_cov.keys():
                cov_values = xds_cov.cov_values.values
                cov_norm = (cov_values - cov_values.mean(axis=0)) / cov_values.std(axis=0)
            else:
                # simulation covars are previously normalized
                cov_norm = xds_cov.cov_norm.values

            # generate covar terms
            for i in range(cov_norm.shape[1]):
                cn = cov_names[i]
                terms[cn] = np.transpose(np.asmatrix(cov_norm[:,i]))
                terms_names.append(cn)

                # Covariates seasonality
                if d_terms_settings['covariates_seasonality'][0]:
                    cov_season = d_terms_settings['covariates_seasonality'][1]

                    if cov_season[i]:
                        terms['{0}_cos'.format(cn)] = np.multiply(
                            terms[cn].T, np.cos(2*np.pi*time_yfrac)
                        ).T
                        terms['{0}_sin'.format(cn)] = np.multiply(
                            terms[cn].T, np.sin(2*np.pi*time_yfrac)
                        ).T
                        terms_names.append('{0}_cos'.format(cn))
                        terms_names.append('{0}_sin'.format(cn))

        # markov term
        if d_terms_settings['mk_order'] > 0:

            # dummi for markov chain
            def dummi(csize):
                D = np.ones((csize-1, csize)) * -1
                for i in range(csize-1):
                    D[i, csize-1-i] = csize-i-1
                    D[i, csize-1+1-i:] = 0
                return D

            def dummi_norm(csize):
                D = np.zeros((csize, csize-1))
                for i in range(csize-1):
                    D[i,i] = float((csize-i-1))/(csize-i)
                    D[i+1:,i] = -1.0/(csize-i)

                return np.transpose(np.flipud(D))

            def helmert_ints(csize, reverse=False):
                D = np.zeros((csize, csize-1))
                for i in range(csize-1):
                    D[i,i] = csize-i-1
                    D[i+1:,i] = -1.0

                if reverse:
                    return np.fliplr(np.flipud(D))
                else:
                    return D

            def helmert_norm(csize, reverse=False):
                D = np.zeros((csize, csize-1))
                for i in range(csize-1):
                    D[i,i] = float((csize-i-1))/(csize-i)
                    D[i+1:,i] = -1.0/(csize-i)

                if reverse:
                    return np.fliplr(np.flipud(D))
                else:
                    return D

            #  helmert
            dum = helmert_norm(cluster_size, reverse=True)

            # solve markov order N
            mk_order = d_terms_settings['mk_order']
            for i in range(mk_order):
                Z = np.zeros((bmus.size, cluster_size-1))
                for indz in range(bmus.size-i-1):
                    Z[indz+i+1,:] = np.squeeze(dum[int(bmus[indz]-1),:])

                terms['markov_{0}'.format(i+1)] = Z

                for ics in range(cluster_size-1):
                    terms_names.append(
                        'mk{0}_{1}'.format(i+1,ics+1)
                    )

        return terms, terms_names

    def GetFracYears(self, time,avgTime):
        'Returns time in custom year decimal format'

        # fix np.datetime64
        if not 'year' in dir(time[0]):
            time_0 = pd.to_datetime(time[0])
            time_1 = pd.to_datetime(time[-1])
            time_d = pd.to_datetime(time[1])
        else:
            time_0 = time[0]
            time_1 = time[-1]
            time_d = time[1]

        # resolution year
        if time_d.year - time_0.year == 1:
            return range(time_1.year - time_0.year+1)

        # resolution day: get start/end data
        y0 = time_0.year
        m0 = time_0.month
        d0 = time_0.day
        y1 = time_1.year
        m1 = time_1.month
        d1 = time_1.day

        # start "year cicle" at 01/01
        d_y0 = date(y0, 1, 1)

        # time array
        d_0 = date(y0, m0, d0)
        d_1 = date(y1, m1, d1)

        # year_decimal from year start to d1
        delta_y0 = d_1 - d_y0
        # y_fraq_y0 = np.array(range(delta_y0.days+1))/365.25
        y_fraq_y0 = np.array(range(int((delta_y0.days)*24/avgTime+24/avgTime)))/(365.25*(24/avgTime))

        # cut year_decimal from d_0
        i0 = int((d_0-d_y0).days*24/avgTime)
        y_fraq = y_fraq_y0[i0:]

        return y_fraq

    def GetFracYearsSWT(self, time,avgTime):
        'Returns time in custom year decimal format'

        # fix np.datetime64
        if not 'year' in dir(time[0]):
            time_0 = pd.to_datetime(time[0])
            time_1 = pd.to_datetime(time[-1])
            time_d = pd.to_datetime(time[1])
        else:
            time_0 = time[0]
            time_1 = time[-1]
            time_d = time[1]

        # resolution year
        if time_d.year - time_0.year == 1:
            return range(time_1.year - time_0.year+1)
        # resolution season
        if time_d.month - time_0.month == 3:
            return np.arange(0,(time_1.year+1)-time_0.year,0.25)
        # resolution day: get start/end data
        y0 = time_0.year
        m0 = time_0.month
        d0 = time_0.day
        y1 = time_1.year
        m1 = time_1.month
        d1 = time_1.day

        # start "year cicle" at 01/01
        d_y0 = date(y0, 1, 1)

        # time array
        d_0 = date(y0, m0, d0)
        d_1 = date(y1, m1, d1)

        # year_decimal from year start to d1
        delta_y0 = d_1 - d_y0
        # y_fraq_y0 = np.array(range(delta_y0.days+1))/365.25
        y_fraq_y0 = np.array(range(int((delta_y0.days)*24/avgTime+24/avgTime)))/365.25

        # cut year_decimal from d_0
        i0 = (d_0-d_y0).days
        y_fraq = y_fraq_y0[i0:]

        return y_fraq

    def FitModel(self, max_iter=1000):
        'Fits ARL model using sklearn'

        # get fitting data
        X = np.concatenate(list(self.terms_fit.values()), axis=1)
        y = self.xds_bmus_fit.bmus.values

        # fit model
        print("\nFitting autoregressive logistic model ...")
        start_time = time.time()

        if self.model_library == 'statsmodels':

            # mount data with pandas
            X = pd.DataFrame(X, columns=self.terms_fit_names)
            y = pd.DataFrame(y, columns=['bmus'])

            # TODO: CAPTURAR LA EVOLUCION DE L (maximun-likelihood)
            self.model = smDD.MNLogit(y,X).fit(
                method='lbfgs',
                maxiter=max_iter,
                retall=True,
                full_output=True,
                disp=True,
                warn_convergence=True,
                missing='raise',
            )

        elif self.model_library == 'sklearn':

            # use sklearn logistig regression
            self.model = linear_model.LogisticRegression(
                penalty='l2', C=1e5, fit_intercept=False,
                solver='lbfgs'
            )
            self.model.fit(X, y)

        else:
            print('wrong config: {0} not in model_library'.format(
                self.model_library
            ))
            sys.exit()

        elapsed_time = time.time() - start_time
        print("Optimization done in {0:.2f} seconds\n".format(elapsed_time))

        # save fitted model
        self.SaveModel()

    def SaveModel(self):
        'Saves fitted model (and fitting terms) for future use'

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        # save model
        pickle.dump(self.model, open(self.p_save_model, 'wb'))

        # save terms fit
        pickle.dump(
            [self.d_terms_settings, self.terms_fit, self.terms_fit_names],
            open(self.p_save_terms_fit, 'wb')
        )

    def LoadModel(self):
        'Load fitted model (and fitting terms)'

        # load model
        self.model = pickle.load(open(self.p_save_model, 'rb'))

        # load terms fit
        self.d_terms_settings, self.terms_fit, self.terms_fit_names = pickle.load(
            open(self.p_save_terms_fit, 'rb')
        )

        # load aux data
        if self.d_terms_settings['covariates'][0]:
            cov_names = self.d_terms_settings['covariates'][1].cov_names.values
            self.cov_names = cov_names

        self.mk_order = self.d_terms_settings['mk_order']

    def SaveBmus_Fit(self):
        'Saves bmus - fit for future use'

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        self.xds_bmus_fit.attrs['cluster_size'] = self.cluster_size
        self.xds_bmus_fit.to_netcdf(self.p_save_fit_xds, 'w')

    def LoadBmus_Fit(self):
        'Load bmus - fit'

        self.xds_bmus_fit =  xr.open_dataset(self.p_save_fit_xds)
        self.cluster_size = self.xds_bmus_fit.attrs['cluster_size']

        return self.xds_bmus_fit

    def SaveBmus_Sim(self):
        'Saves bmus - sim for future use'

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        self.xds_bmus_sim.to_netcdf(self.p_save_sim_xds, 'w')

    def LoadBmus_Sim(self):
        'Load bmus - sim'

        self.xds_bmus_sim =  xr.open_dataset(self.p_save_sim_xds)

        return self.xds_bmus_sim

    def Report_Fit(self, terms_fit=False, summary=False, show=True):
        'Report containing model fitting info'

        # load model
        self.LoadModel()

        # get data
        try:
            pval_df = self.model.pvalues.transpose()
            params_df = self.model.params.transpose()
            name_terms = pval_df.columns.tolist()
        except:
            # TODO: no converge?
            print('warning - statsmodels MNLogit could not provide p-values')
            return

        # output figs
        l_figs = []

        # plot p-values
        f = Plot_PValues(pval_df.values, name_terms, show=show)
        l_figs.append(f)

        # plot parameters
        f = Plot_Params(params_df.values, name_terms, show=show)
        l_figs.append(f)

        # plot terms used for fitting
        if terms_fit:
            f = self.Report_Terms_Fit(p_rep_trms, show=show)
            l_figs.append(f)

        # write summary
        if summary:
            summ = self.model.summary()
            print(summ.as_text())

    def Report_Terms_Fit(self, show=True):
        'Plot terms used for model fitting'

        # load bmus fit
        self.LoadBmus_Fit()

        # get data for plotting
        term_mx = np.concatenate(list(self.terms_fit.values()), axis=1)
        term_ds = [npdt64todatetime(t) for t in self.xds_bmus_fit.time.values]
        term_ns = self.terms_fit_names

        # Plot terms
        f = Plot_Terms(term_mx, term_ds, term_ns, show=show)
        return f

    def Simulate(self, num_sims, time_sim, xds_covars_sim=None,
                 log_sim=False, overfit_filter=False, of_probs=0.98, of_pers=5,avgTime=None):
        '''
        Launch ARL model simulations

        num_sims           - number of simulations to compute
        time_sim           - time array to solve

        xds_covars_sim     - xr.Dataset (time,), cov_values
            Covariates used at simulation, compatible with "n_sim" dimension
            ("n_sim" dimension (optional) will be iterated with each simulation)

        log_sim            - Store a .nc file with all simulation detailed information.

        filters for exceptional ALR overfit probabilities situation and patch:

        overfit_filter     - overfit filter activation
        of_probs           - overfit filter probabilities activation
        of_pers            - overfit filter persistences activation
        '''

        class SimLog(object):
            '''
            simulatiom log - records and stores detail info for each time step and n_sim
            '''
            def __init__(self, time_yfrac, mk_order, num_sims, cluster_size, terms_fit_names):

                # initialize variables to record
                self.terms = np.nan * np.zeros((len(time_yfrac)-mk_order,
                                                num_sims, mk_order+1, len(terms_fit_names)))
                self.probs = np.nan * np.zeros((len(time_yfrac)-mk_order, num_sims, mk_order+1, cluster_size))
                self.ptrns = np.nan * np.zeros((len(time_yfrac)-mk_order, num_sims, cluster_size))
                self.nrnd = np.nan * np.zeros((len(time_yfrac)-mk_order, num_sims))
                self.evbmu_sims = np.nan * np.zeros((len(time_yfrac)-mk_order, num_sims))

                # overfit filter variables (filter states and filtered bmus)
                self.of_state = np.zeros((len(time_yfrac)-mk_order, num_sims), dtype=bool)
                self.of_bmus = np.nan * np.zeros((len(time_yfrac)-mk_order, num_sims))

            def Add(self, ix_t, ix_s, terms, prob, probTrans, nrnd, of_state, of_bmus):

                # add iteration to log
                self.terms[ix_t,ix_s,:,:] = terms
                self.probs[ix_t,ix_s,:,:] = prob
                self.ptrns[ix_t,ix_s,:] = probTrans
                self.nrnd[ix_t,ix_s] = nrnd
                self.evbmu_sims[ix_t,ix_s] = np.where(probTrans>nrnd)[0][0]+1

                # add overfit filter data to log
                self.of_state[ix_t,ix_s] = of_state
                self.of_bmus[ix_t,ix_s] = of_bmus

            def Save(self, p_save, terms_names):

                # use xarray to store netcdf
                xds_log = xr.Dataset(
                    {
                        'alr_terms': (('time', 'n_sim', 'mk', 'terms',), self.terms),
                        'probs': (('time', 'n_sim', 'mk', 'n_clusters'), self.probs),
                        'probTrans': (('time', 'n_sim', 'n_clusters'), self.ptrns),
                        'nrnd': (('time', 'n_sim'), self.nrnd),
                        'evbmus_sims': (('time', 'n_sim'), self.evbmu_sims.astype(int)),

                        'overfit_filter_state': (('time', 'n_sim'), self.of_state),
                        'evbmus_sims_filtered': (('time', 'n_sim'), self.of_bmus.astype(int)),
                    },

                    coords = {
                        'time' : time_sim[mk_order:],
                        'terms' : terms_names,
                    },
                )

                StoreBugXdset(xds_log, p_save)
                print('simulation data log stored at {0}\n'.format(p_save))

        class OverfitFilter(object):
            '''
            overfit filter for alr outlayer outputs.
            '''
            def __init__(self, probs_lim, pers_lim):

                self.active = False
                self.probs_lim = probs_lim
                self.pers_lim = pers_lim
                self.log = ''

            def CheckStatus(self, n_sim, prob, bmus):
                'check current iteration filter status'

                # active filter
                if self.active:

                    # continuation condition
                    self.active = np.nanmax(prob[-1, :]) >= self.probs_lim

                    # log when deactivated
                    if self.active == False:
                        self.log += 'sim. {0:02d} - {1} - deactivated (max prob {2})\n'.format(
                            n_sim, time_sim[i], np.nanmax(prob[-1,:]))

                # inactive filter
                else:

                    # re-activation condition
                    self.active = np.nanmax(prob[-1, :]) >= self.probs_lim and \
                            np.all(evbmus[-1*self.pers_lim:]==new_bmus)

                    # log when activated
                    if self.active:
                        self.log += 'sim. {0:02d} - {1} - activated (max prob {2})\n'.format(
                            n_sim, time_sim[i], np.nanmax(prob[-1,:]))

            def PrintLog(self):
                'Print filter log'

                if self.log != '':
                    print('overfit filter log')
                    print(self.log)


        # switch library probabilities predictor function
        if self.model_library == 'statsmodels':
            pred_prob_fun = self.model.predict
        elif self.model_library == 'sklearn':
            pred_prob_fun = self.model.predict_proba
        else:
            print('wrong config: {0} not in model_library'.format(
                self.model_library
            ))
            sys.exit()

        # get needed data
        evbmus_values = self.xds_bmus_fit.bmus.values
        time_fit = self.xds_bmus_fit.time.values
        mk_order = self.mk_order
        print('{},{},{}'.format(time_fit[0],time_fit[1],time_fit[2]))

        # times at datetime
        if isinstance(time_sim[0], np.datetime64):
            time_sim = [npdt64todatetime(t) for t in time_sim]
        print('{},{},{}'.format(time_sim[0],time_sim[1],time_sim[2]))

        # print some info
        tf0 = str(time_fit[0])[:10]
        tf1 = str(time_fit[-1])[:10]
        ts0 = str(time_sim[0])[:10]
        ts1 = str(time_sim[-1])[:10]
        print('ALR model fit   : {0} --- {1}'.format(tf0, tf1))
        print('ALR model sim   : {0} --- {1}'.format(ts0, ts1))
        # generate time yearly fractional array
        time_yfrac = self.GetFracYears(time_sim,avgTime)
        # time_yfrac = self.GetFracYearsSWT(time_sim)

        print('{},{},{}'.format(time_yfrac[0],time_yfrac[1],time_yfrac[2]))

        # use a d_terms_settigs copy
        d_terms_settings_sim = self.d_terms_settings.copy()

        # initialize optional simulation log
        if log_sim:
            SL = SimLog(time_yfrac, mk_order, num_sims, self.cluster_size, self.terms_fit_names)

        # initialize ALR overfit filter
        ofilt = OverfitFilter(of_probs, of_pers)

        # initialize ALR simulated bmus array, and overfit filter register array
        evbmus_sims = np.zeros((len(time_yfrac), num_sims))
        ofbmus_sims = np.zeros((len(time_yfrac), num_sims), dtype=bool)

        # start simulations
        print("\nLaunching {0} simulations...\n".format(num_sims))
        for n in range(num_sims):

            # preload some data (simulation covariates)
            cvtxt = ''
            if xds_covars_sim != None:

                # check if n_sim dimension in xds_covars_sim
                if 'n_sim' in xds_covars_sim.dims:
                    sim_covars_T = xds_covars_sim.isel(n_sim=n).cov_values.values
                    cvtxt = ' (Covs. {0:03d})'.format(n+1)
                else:
                    sim_covars_T = xds_covars_sim.cov_values.values

                sim_covars_T_mean = sim_covars_T.mean(axis=0)
                sim_covars_T_std = sim_covars_T.std(axis=0)

            # progress bar
            pbar = tqdm(
                total=len(time_yfrac)-mk_order,
                file=sys.stdout,
                desc = 'Sim. Num. {0:03d}{1}'.format(n+1, cvtxt)
            )

            evbmus = evbmus_values[1:mk_order+1]
            for i in range(len(time_yfrac) - mk_order):

                # handle simulation covars
                if d_terms_settings_sim['covariates'][0]:

                    # normalize step covars
                    sim_covars_evbmus = sim_covars_T[i : i + mk_order +1]
                    sim_cov_norm = (sim_covars_evbmus - sim_covars_T_mean
                                    ) / sim_covars_T_std

                    # mount step xr.dataset for sim covariates
                    xds_cov_sim_step = xr.Dataset(
                        {
                            'cov_norm': (('time','cov_names'), sim_cov_norm),
                        },
                        coords = {'cov_names': self.cov_names}
                    )

                    d_terms_settings_sim['covariates'] = (True, xds_cov_sim_step)

                # generate time step ALR terms
                terms_i, terms_names = self.GenerateALRTerms(
                    d_terms_settings_sim,
                    np.append(evbmus[ i : i + mk_order], 0),
                    time_yfrac[i : i + mk_order + 1],
                    self.cluster_size, avgTime, time2yfrac=False)

                # Event sequence simulation  (sklearn)
                X = np.concatenate(list(terms_i.values()), axis=1)
                prob = pred_prob_fun(X)  # statsmodels // sklearn functions
                probTrans = np.cumsum(prob[-1,:])

                # generate random cluster with ALR probs
                nrnd = np.random.rand()
                new_bmus = np.where(probTrans>nrnd)[0][0]+1

                # overfit filter status swich (if active)
                if overfit_filter:
                    ofilt.CheckStatus(n, prob, np.append(evbmus, new_bmus))

                # override overfit bmus if filter active
                if ofilt.active:
                    # criteria: random bmus from that date of the year at  historical
                    ix_of = np.random.choice(np.where(
                        self.xds_bmus_fit["time.dayofyear"] == time_sim[i].timetuple().tm_yday)[0])
                    new_bmus = self.xds_bmus_fit.bmus.values[ix_of]

                # append_bmus
                evbmus = np.append(evbmus, new_bmus)

                # store overfit filter status
                ofbmus_sims[i+mk_order, n] = ofilt.active

                # optional detail log
                if log_sim: SL.Add(i, n, X, prob, probTrans, nrnd, ofilt.active, new_bmus)

                # update progress bar
                pbar.update(1)

            evbmus_sims[:,n] = evbmus

            # close progress bar
            pbar.close()
        print()  # white line after all progress bars

        # return ALR simulation data in a xr.Dataset
        xds_out = xr.Dataset(
            {
                'evbmus_sims': (('time', 'n_sim'), evbmus_sims.astype(int)),
                'ofbmus_sims': (('time', 'n_sim'), ofbmus_sims),
            },

            coords = {
                'time' : time_sim,
            },
        )

        # save output
        StoreBugXdset(xds_out, self.p_save_sim_xds)

        # save log file
        if log_sim: SL.Save(self.p_log_sim_xds, terms_names)

        # overfit filter log
        ofilt.PrintLog()

        return xds_out

    def Report_Sim(self, py_month_ini=1, persistences_hists=False, persistences_table=False, show=True):
        '''
        Report that Compare fitting to simulated bmus

        py_month_ini  - start month for PerpetualYear bmus comparison
        '''
        # TODO: add arg n_sim = None (for plotting only one sim output)

        # load fit and sim bmus
        xds_ALR_fit = self.LoadBmus_Fit()
        xds_ALR_sim = self.LoadBmus_Sim()

        # report folder and files
        p_save = self.p_report_sim

        # get data
        cluster_size = self.cluster_size
        bmus_values_sim = xds_ALR_sim.evbmus_sims.values[:]
        bmus_dates_sim = xds_ALR_sim.time.values[:]
        bmus_values_hist = np.reshape(xds_ALR_fit.bmus.values,[-1,1])
        bmus_dates_hist = xds_ALR_fit.time.values[:]
        num_sims = bmus_values_sim.shape[1]

        # calculate bmus persistences
        pers_hist = Persistences(bmus_values_hist.flatten())
        lsp = [Persistences(bs) for bs in bmus_values_sim.T.astype(int)]
        pers_sim = {k:np.concatenate([x[k] for x in lsp]) for k in lsp[0].keys()}

        # fix datetime 64 dates
        if isinstance(bmus_dates_sim[0], np.datetime64):
            bmus_dates_sim = [npdt64todatetime(t) for t in bmus_dates_sim]
        if isinstance(bmus_dates_hist[0], np.datetime64):
            bmus_dates_hist = [npdt64todatetime(t) for t in bmus_dates_hist]

        # output figs
        l_figs = []

        # Plot Perpetual Year (daily) - bmus wt
        fig_PP = Plot_Compare_PerpYear(
            cluster_size,
            bmus_values_sim, bmus_dates_sim,
            bmus_values_hist, bmus_dates_hist,
            n_sim = num_sims, month_ini=py_month_ini,
            show = show,
        )
        l_figs.append(fig_PP)

        # Plot WTs Transition (probability change / scatter Fit vs. ACCUMULATED Sim) 
        sttl = 'Cluster Probabilities Transitions: All Simulations'
        fig_CT = Plot_Compare_Transitions(
            cluster_size, bmus_values_hist, bmus_values_sim,
            sttl = sttl, show = show,
        )
        l_figs.append(fig_CT)


        # Plot Persistences comparison Fit vs Sim
        if persistences_hists:
            fig_PS = Plot_Compare_Persistences(
                cluster_size,
                pers_hist, pers_sim,
                show = show,
            )
            l_figs.append(fig_PS)

        # persistences set table
        if persistences_table:
            print('Persistences by WT (set)')
            for c in range(cluster_size):
                wt=c+1
                p_h = pers_hist[wt]
                p_s = pers_sim[wt]

                print('WT: {0}'.format(wt))
                print('  hist : {0}'.format((sorted(set(p_h)))))
                print('  sim. : {0}'.format(sorted(set(p_s))))


        # TODO export handling (if show=False)
        #p_save = self.p_report_sim
        #p_rep_PY = None
        #p_rep_VL = None
        #if export:
        #    if not op.isdir(p_save): os.mkdir(p_save)
        #    p_rep_PY = op.join(p_save, 'PerpetualYear.png')
        #    p_rep_VL = op.join(p_save, 'Transitions.png')

        return l_figs

        # TODO: Plot Perpetual Year (monthly)
        # TODO: load covariates if needed for ploting
        # TODO: activate this
        if self.d_terms_settings['covariates'][0]:

            # TODO eliminar tiempos duplicados
            time_hist_covars = bmus_dates_hist
            time_sim_covars = bmus_dates_sim

            # covars fit
            xds_cov_fit = self.d_terms_settings['covariates'][1]
            cov_names = xds_cov_fit.cov_names.values
            cov_fit_values = xds_cov_fit.cov_values.values

            # covars sim
            cov_sim_values = xds_cov_sim.cov_values.values

            for ic, cn in enumerate(cov_names):

                # get covariate data
                cf_val = cov_fit_values[:,ic]
                cs_val = cov_sim_values[:,ic]

                # plot covariate - bmus wt
                p_rep_cn = op.join(p_save, '{0}_comp.png'.format(cn))
                Plot_Compare_Covariate(
                    cluster_size,
                    bmus_values_sim, bmus_dates_sim,
                    bmus_values_hist, bmus_dates_hist,
                    cs_val, time_sim_covars,
                    cf_val, time_hist_covars,
                    cn,
                    n_sim = num_sims, p_export = p_rep_cn
                )

    def Report_Sim_Log(self, n_sim=0, t_slice=None, show=True):
        '''
        Interactive plot for simulation log

        n_sim  - simulation log to plot
        '''

        # load fit and sim bmus
        xds_log = xr.open_dataset(self.p_log_sim_xds, decode_times=True)

        # get simulation
        log_sim = xds_log.isel(n_sim=n_sim)

        if t_slice != None:
            log_sim = log_sim.sel(time=t_slice)

        # plot interactive report
        Plot_Log_Sim(log_sim);





def running_mean(x, N, mode_str='mean'):
    '''
    computes a running mean (also known as moving average)
    on the elements of the vector X. It uses a window of 2*M+1 datapoints

    As always with filtering, the values of Y can be inaccurate at the
    edges. RUNMEAN(..., MODESTR) determines how the edges are treated. MODESTR can be
    one of the following strings:
      'edge'    : X is padded with first and last values along dimension
                  DIM (default)
      'zeros'   : X is padded with zeros
      'ones'    : X is padded with ones
      'mean'    : X is padded with the mean along dimension DIM

    X should not contains NaNs, yielding an all NaN result.
    '''

    # if nan in data, return nan array
    if np.isnan(x).any():
        return np.full(x.shape, np.nan)

    nn = 2*N+1

    if mode_str == 'zeros':
        x = np.insert(x, 0, np.zeros(N))
        x = np.append(x, np.zeros(N))

    elif mode_str == 'ones':
        x = np.insert(x, 0, np.ones(N))
        x = np.append(x, np.ones(N))

    elif mode_str == 'edge':
        x = np.insert(x, 0, np.ones(N)*x[0])
        x = np.append(x, np.ones(N)*x[-1])

    elif mode_str == 'mean':
        x = np.insert(x, 0, np.ones(N)*np.mean(x))
        x = np.append(x, np.ones(N)*np.mean(x))


    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)


def RunnningMean_Monthly(xds, var_name, window=5):
    '''
    Calculate running average grouped by months

    xds:
        (longitude, latitude, time) variables: var_name

    returns xds with new variable "var_name_runavg"
    '''

    tempdata_runavg = np.empty(xds[var_name].shape)

    for lon in xds.longitude.values:
       for lat in xds.latitude.values:
          for mn in range(1, 13):

             # indexes
             ix_lon = np.where(xds.longitude == lon)
             ix_lat = np.where(xds.latitude == lat)
             ix_mnt = np.where(xds['time.month'] == mn)

             # point running average
             time_mnt = xds.time[ix_mnt]
             data_pnt = xds[var_name].loc[lon, lat, time_mnt]

             tempdata_runavg[ix_lon[0], ix_lat[0], ix_mnt[0]] = running_mean(
                 data_pnt.values, window)

    # store running average
    xds['{0}_runavg'.format(var_name)]= (
        ('longitude', 'latitude', 'time'),
        tempdata_runavg)

    return xds

def PCA_LatitudeAverage(xds, var_name, y1, y2, m1, m2):
    from sklearn.decomposition import PCA

    '''
    Principal component analysis
    method: remove monthly running mean and latitude average

    xds:
        (longitude, latitude, time), pred_name | pred_name_runavg

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # calculate monthly running mean
    xds = RunnningMean_Monthly(xds, var_name)

    # predictor variable and variable_runnavg from dataset
    var_val = xds[var_name]
    var_val_ra = xds['{0}_runavg'.format(var_name)]

    # use datetime for indexing
    dt1 = datetime(y1, m1, 1)
    dt2 = datetime(y2+1, m2, 28)
    time_PCA = [datetime(y, m1, 1) for y in range(y1, y2+1)]

    # use data inside timeframe
    data_ss = var_val.loc[:,:,dt1:dt2]
    data_ss_ra = var_val_ra.loc[:,:,dt1:dt2]

    # anomalies: remove the monthly running mean
    data_anom = data_ss - data_ss_ra

    # average across all latitudes
    data_avg_lat = data_anom.mean(dim='latitude')

    # collapse 12 months of data to a single vector
    nlon = data_avg_lat.longitude.shape[0]
    ntime = data_avg_lat.time.shape[0]
    hovmoller = xr.DataArray(
        np.reshape(data_avg_lat.values, (12*nlon, ntime//12), order='F')
    )
    hovmoller = hovmoller.transpose()

    # mean and standard deviation
    var_anom_mean = hovmoller.mean(axis=0)
    var_anom_std = hovmoller.std(axis=0)

    # remove means and normalize by the standard deviation at anomaly
    # rows = time, columns = longitude
    nk_m = np.kron(np.ones((y2-y1+1,1)), var_anom_mean)
    nk_s = np.kron(np.ones((y2-y1+1,1)), var_anom_std)
    var_anom_demean = (hovmoller - nk_m) / nk_s

    # sklearn principal components analysis
    ipca = PCA(n_components=var_anom_demean.shape[0])
    PCs = ipca.fit_transform(var_anom_demean)

    pred_lon = xds.longitude.values[:]
    # print(ipca)
    # print(var_anom_std)
    # print(var_anom_mean)
    return xr.Dataset(
        {
            'PCs': (('n_components', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'var_anom_std': (('n_features',), var_anom_std.values),
            'var_anom_mean': (('n_features',), var_anom_mean.values),

            'time': (('n_components'), time_PCA),
            'pred_lon': (('n_lon',), pred_lon),
        },

        # store PCA algorithm metadata
        attrs = {
            'method': 'anomalies, latitude averaged',
        }
    )



def KMA_simple(xds_PCA, num_clusters, repres=0.95):
    '''
    KMeans Classification for PCA data

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance
    num_clusters
    repres

    returns a xarray.Dataset containing KMA data
    '''
    from sklearn.cluster import KMeans

    # PCA data
    variance = xds_PCA.variance.values[:]
    EOFs = xds_PCA.EOFs.values[:]
    PCs = xds_PCA.PCs.values[:]

    var_anom_std = xds_PCA.var_anom_std.values[:]
    var_anom_mean = xds_PCA.var_anom_mean.values[:]
    time = xds_PCA.time.values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs[:, :nterm+1]
    EOFsub = EOFs[:nterm+1, :]

    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)

    # groupsize
    _, group_size = np.unique(kma.labels_, return_counts=True)

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_==k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET

    # centroids
    centroids = np.dot(kma.cluster_centers_, EOFsub)

    # km, x and var_centers
    km = np.multiply(
        centroids,
        np.tile(var_anom_std, (num_clusters, 1))
    ) + np.tile(var_anom_mean, (num_clusters, 1))

    # sort kmeans
    kma_order = np.argsort(np.mean(-km, axis=1))

    # reorder clusters: bmus, km, cenEOFs, centroids, group_size
    sorted_bmus = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_ == kma_order[i])
        sorted_bmus[posc] = i
    sorted_km = km[kma_order]
    sorted_cenEOFs = kma.cluster_centers_[kma_order]
    sorted_centroids = centroids[kma_order]
    sorted_group_size = group_size[kma_order]

    return xr.Dataset(
        {
            'bmus': (('n_pcacomp'), sorted_bmus.astype(int)),
            'cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'centroids': (('n_clusters','n_pcafeat'), sorted_centroids),
            'Km': (('n_clusters','n_pcafeat'), sorted_km),
            'group_size': (('n_clusters'), sorted_group_size),

            # PCA data
            'PCs': (('n_pcacomp','n_features'), PCsub),
            'variance': (('n_pcacomp',), variance),
            'time': (('n_pcacomp',), time),
        }
    )
