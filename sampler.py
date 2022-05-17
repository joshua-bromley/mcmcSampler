#Preamble
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from multiprocessing import Pool
from multiprocessing import cpu_count
import lightkurve as lk
import pandas as pd
import math
import time
import copy
import pickle
import emcee
import corner
import os
import scipy

os.environ["OMP_NUM_THREADS"] = "1"

'''
The following functions are for simulating a transit.
generateEllipse, transitSim
The transit simulation assumes that the occulter is an ellipse and the star is a circle. 
These shapes are discritized into pixels where one pizel is the movement of the occulter over the minimum time step or the star is 50 pixels in radius, whichever is higher resolution
'''

def generateEllipse(a,b,centerX,centerY, grid, opacity):
    '''
    Creates a two dimensional array with an ellipse filled in
    Used by transitSim
    Parameters:
    a: The width of the ellipse
    b: The height of the ellipse
    centerX: The x coordinate of the center of the ellipse, must be larger than a
    centerY: The y coordinate of the center of the ellipse, mast be larger than b
    grid: The two dimennsional array to generate the ellipse on, must be larger than [centerX + a][centerY + b]
    opacity: Value to give the filled in values of the ellipse

    Returns:
    grid: the filled in grid with an ellipse
    '''
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if a > 0 and b > 0:
                if ((x-centerX)/a)*((x-centerX)/a) + ((y-centerY)/b)*((y-centerY)/b) <= 1:
                    grid[x][y] = opacity
    
    return grid

def transitSim(a,b,r, speed, times, tref, opacity, impact):
    '''
    Creates a light curve of a transiting ellipse from the paramters
    Used by logLikelihood and to plot the light curves
    Parameters:
    a: The width of the ellipse
    b: The height of the ellipse
    r: The radius of the star
    speed: The speed at which the ellipse transits (in stellar radii per unit time)
    times: A list of time points to generate a value for
    tref: The location of the center of the transit in the time array by value
    opacity: The opacity of the ellipse, although there is no technical limit for this value, the physical limit is [0,1] 
    impact: Impact parameter of the occulter
    '''
    ##Calculate the dimension ratios incase the ellipse has to be resized
    ab = a/b
    br = b/r
    
    tmin = tref - 1/(2*speed) ##Calculate the lower time bound for the transit
    tmax = tref + 1/(2*speed) ##Calculate the upper time bound for the transit
    transitTimes = [t for t in times if t >= tmin and t <= tmax] ##Create an array of only the times included in the transit
    flux = [1 for t in times if t < tmin] ##Create an array of the flux values, setting it to 1 before the transit
    differences = [] ##Calculate the differences between the time points
    for i in range(1,len(transitTimes)):
        differences.append(transitTimes[i] - transitTimes[i-1])
    
    if len(differences) == 0:
        differences.append(times[1] - times[0])
    minDiff = np.min(differences)
    intDiffs = [int(d/minDiff) for d in differences]##Normalize the difference so the smallest one has value unity
    length = np.sum(intDiffs)##Calculate the length of the transit in steps
    
    
    res = int((2*(b+r))/length) ##Resize the dimensions so the length matches what is required for the time array
    if res < 1:
        res = 1
    rnew = int(res*length/(2*(1+br)))
    bnew = int(br*rnew)
    anew = int(ab*bnew)
    newIntDiffs = [res*i for i in intDiffs]
    
    starGrid = np.zeros([2*rnew+impact, 4*bnew+2*rnew+4])
    ellipseGrid = np.zeros([2*rnew+impact, 4*bnew+2*rnew+4])
    starGrid = generateEllipse(rnew,rnew,rnew, 2*bnew+rnew, starGrid,1)##Generate the star
    ellipseGrid = generateEllipse(anew,bnew,rnew+impact,3*bnew+2*rnew + 2,ellipseGrid,opacity)##Genrate the occulter
    planetGrid = np.ones([2*rnew+impact,4*bnew+2*rnew+ 4]) - ellipseGrid
    fluxGrid = np.multiply(starGrid,planetGrid)##Calculate the flux by item by item multiplying each pixel in the star and planet grid and summing
    initialFlux = np.sum(fluxGrid)
    
    for i in newIntDiffs:
        for j in range(i):
            planetGrid = np.delete(planetGrid,0,1)
            planetGrid = np.append(planetGrid,np.ones([2*rnew+impact,1]),1)##Move the first column to the end to "move" the planet across the star
            
        
        fluxGrid = np.multiply(starGrid,planetGrid)
        percentFlux = np.sum(fluxGrid)/initialFlux
        flux.append(percentFlux)#Calculate a normalized value of flux
        
    for t in times:
        if t > tmax:
            flux.append(1)##Add full flux after the transit
    
    flux.append(1)
    return flux


'''
The following fucntions are used in the MCMC sampler
logLikelihood, logPrior, logProbability
'''

def logLikelihood(theta, times, flux, fluxErr):
    """
    Calculates the log likelihood based on the difference between the model and the data
    Used by logProbability
    Args:
        theta (list) - parameters of the model
        times (list) - time array of the light curve
        flux (list) - array of flux data points
        fluxErr (list) - array of errors for the flux data points

    Returns:
        lnl (float) - log likelihood for the given theta values
    """
    xdim, ydim, velocity, opacity, impact, tRef = theta
    fluxPredicted = transitSim(ydim, xdim,50,velocity,times, tRef,opacity,int(impact)) ##Simulates a transit to evaluate the parameters
    error = [((flux[i] - fluxPredicted[i])**2) /(2*fluxErr[i]**2) for i in range(len(flux))] ##Calcutes Chi Squared error between model and data
    lnl = -np.sum(error)
    return lnl

def logPrior(theta, times, minFlux):
    """
    Returns flat priors, checking that the given theta values are physically possible
    Used by logProbability
    Args:
        theta (list) - parameters of the model
        times (list) time array of the light curve
        
    Returns: 
        lnPrior (float) - fixed log prior value if theta values are allowed, -inf if theta values aren't
    """
    xdim, ydim, velocity, opacity,impact, tRef = theta
    lnPrior = 0
    if 0 < xdim < 50 and 0 < ydim < 50 and 0 < velocity < 50 and 0 < opacity < 1.01 and 0 < impact < 100 and times[0] < tRef < times[-1]: ##Check to see if the shape exists but is not larger than the star
        ##Also check to see that it transits in a consistent direction and not extremely fast and that it is not super dark
        lnPrior +=  + np.log(1/50) + np.log(1)
    else:
        return -np.inf

    predictedRadius = 50*np.sqrt(1-minFlux) ##Predicts radius based on a circle
    lnPrior += (5000/(np.sqrt(2*3.1415926535)*5))*np.exp(-0.5*((xdim-predictedRadius)/5)**2) ##Prior encourages to object to be round
    lnPrior += (5000/(np.sqrt(2*3.1415926535)*5))*np.exp(-0.5*((ydim-predictedRadius)/5)**2)
    return lnPrior

def logProbability(theta, times, flux, fluxErr,minFlux):
    """
    Combines the log likelihood and log prior to get log probability
    Used by emcee.sample()
    Args:
        theta (list) - parameters of the model
        times (list)
    """
    startTime = time.time()
    lp = logPrior(theta, times,minFlux)
    if not np.isfinite(lp):
        return -np.inf
    ll = logLikelihood(theta, times, flux, fluxErr)
    endTime = time.time()
    #print(endTime - startTime)
    return (lp + ll)




'''The following functions are used for processing the light curve before sampling
    load_lc, cutLightCurve
    load_lc is written by and borrowed from Daniel Giles https://github.com/d-giles/SPOcc
'''

def load_lc(fp, fluxtype="PDC", mask=False):
    """Load light curve data from pickle file into a lightkurve object
    Args:
        fp (str) - file path to pickle file in standard format
        fluxtype (str) - Type of flux to prioritize,
            choose between "raw", "corr", and "PDC"
        mask (bool) - Mask data points non-zero flags in quality

    returns:
        lc (lightkurve.lightcurve.LightCurve) - a LightCurve object
    """

    with open(fp, 'rb') as file:
        lc_list = pickle.load(file)

    fluxes = {"raw": lc_list[7], "corr": lc_list[8], "PDC": lc_list[9]}

    try:
        flux = fluxes[fluxtype]

    except KeyError:
        print("""
        The flux type must be 'raw', 'corr', or 'PDC'. Defaulting to 'PDC'.""")
        flux = fluxes["PDC"]

    finally:
        time = lc_list[6]
        flux_err = lc_list[10]
        quality = lc_list[11]

        if mask:
            mask = lc_list[11] == 0
            flux = flux[mask]
            time = time[mask]
            flux_err = flux_err[mask]
            quality = quality[mask]  # just 0's if masked

        # for meta information
        fluxes.update(
            {"TESS Magnitude": lc_list[3], "filename": fp.split("/")[-1]})
        lc = lk.lightcurve.TessLightCurve(
            time=time, flux=flux, flux_err=flux_err, targetid=lc_list[0],
            quality=quality, camera=lc_list[4], ccd=lc_list[5],
            ra=lc_list[1], dec=lc_list[2], label=f"TIC {lc_list[0]}",
            meta=fluxes
        )

    return lc

def cutLightCurve(times, lc, err, t0, t1):
    '''
    Truncates light curves using the mjd units
    Parameters:
        times: The array of time valeus
        lc: The array of fluxes
        err: The array of errors
        t0: The initial time (cuts everything before)
        t1: The final time (cuts everything after)
    Returns:
        newTimes: Cut times arrar
        newLc: Cut flux array
        newErr: Cut error array
    '''
    newTimes = [] ##Create arrays for cur light curve after its cut
    newLc = []
    newErr = []
    for i in range(len(times)):
        if times[i] > t0 and times[i] < t1: ##Iterating through the time array, keep only the datapoint between the t0 and t1
            newTimes.append(times[i])
            newLc.append(lc[i])
            newErr.append(err[i])
            
    return newTimes,newLc, newErr

'''
The following functions are used for phase folding light curves before processing.
fold, redef, calcresidualstddevmin, calcresidualstddevmax, calcresidualstddevmidpt
These functions are written by and borrowed from Yifan Tong https://github.com/yifantong1010/PhaseFold
'''

def fold(lcurve):
    '''
    Phase folds the given light curve and outputs 3 graphs: the original light curve, the periodogram, and the folded light curve
    As well as gives the option to save the 3 graphs and the combined graph in the LightCurves folder.
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    lc = lightc[lightc.quality==0]
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    #cleanlightcurve = folded[folded.quality==0]
    return folded

def redef(mini, maxi, midpt, lc):
    '''
    Adjusts the midpt after comparing the standard deviation of the residuals to find the best period
    
        Parameters:
            mini (double): current minimum period for binary search
            maxi (double): current maximum period for binary search
            midpt (double): current assumed period value
            lc (lightkurve.LightCurve): a lightkurve.LightCurve object
        
        Returns: 
            midpt (double): the best period to phase fold on
    '''
    #tests if a multiple of the midpt is better than the current one
    twomidptresstddev = calcresidualstddevmidpt(lc, 2*midpt)
    midptresstddev = calcresidualstddevmidpt(lc, midpt)
    
    if (twomidptresstddev) < (midptresstddev):
        midpt = 2*midpt
        mini = .7*midpt
        maxi = 1.3*midpt
    
    #global mini, maxi, midpt, lc
    while(mini.value + 0.0001 < maxi.value):
        minresstddev = calcresidualstddevmin(lc, mini)
        midptresstddev = calcresidualstddevmidpt(lc, midpt)
        maxresstddev = calcresidualstddevmax(lc, maxi)
        if (minresstddev - midptresstddev) < (maxresstddev - midptresstddev):
            maxi = midpt
            midpt = (mini + maxi)/2
        else:
            mini = midpt
            midpt = (mini + maxi)/2
    
    return midpt

def calcresidualstddevmin(lc, num):
    lcfolded = lc.fold(num)
    cleanlightcurve = lcfolded[lcfolded.quality==0]
    phasecurve = lc.fold(num)[:]
    cleanlcmod = cleanlightcurve[:]
    cleanlcmod.flux = scipy.signal.medfilt(cleanlightcurve.flux, kernel_size=13)
    residual = cleanlcmod.flux.value - cleanlightcurve.flux.value
    ressqr = 0
    for x in range(len(cleanlcmod)):
        ressqr = ressqr + (residual[x] ** 2)
    minresstddev = math.sqrt((ressqr)/(len(cleanlcmod)-2))
    return minresstddev

def calcresidualstddevmax(lc, num):
    lcfolded = lc.fold(num)
    cleanlightcurve = lcfolded[lcfolded.quality==0]
    phasecurve = lc.fold(num)[:]
    cleanlcmod = cleanlightcurve[:]
    cleanlcmod.flux = scipy.signal.medfilt(cleanlightcurve.flux, kernel_size=13)
    residual = cleanlcmod.flux.value - cleanlightcurve.flux.value
    ressqr = 0
    for x in range(len(cleanlcmod)):
        ressqr = ressqr + (residual[x] ** 2)
    maxresstddev = math.sqrt((ressqr)/(len(cleanlcmod)-2))
    return (maxresstddev)
    
def calcresidualstddevmidpt(lc, num):
    lcfolded = lc.fold(num)
    cleanlightcurve = lcfolded[lcfolded.quality==0]
    phasecurve = lc.fold(num)[:]
    cleanlcmod = cleanlightcurve[:]
    cleanlcmod.flux = scipy.signal.medfilt(cleanlightcurve.flux, kernel_size=13)
    residual = cleanlcmod.flux.value - cleanlightcurve.flux.value
    ressqr = 0
    for x in range(len(cleanlcmod)):
        ressqr = ressqr + (residual[x] ** 2)
    midptresstddev = math.sqrt((ressqr)/(len(cleanlcmod)-2))
    return (midptresstddev)


'''
##Code for pulling a light curve from lightkurve
searchResult = lk.search_lightcurve("KIC 8462852", quarter = 8)
lc = searchResult.download()
'''
mountDir = '/mnt/disks/lcs/' ##This is the directory where the light curve disk is mounted

tics = ["126944775","349480507","139699256"] ##This array is the TIC Ids of all the light curves to fit
sectors = ["1","1","1"] ##This array is the sector correcsponding to the sectors for the TIC IDs in tics. Each entry in tics must have a corresponding entry in T
for i in range(len(tics)):
    tic = tics[i]
    ##Before running this for the first time since turning on the machine run sudo mount -o discard,ro /dev/[diskname] [mount Directory] in the terminal
    ##Disks are usually named sd[b,c,e ...]
    ref = pd.read_csv(mountDir + "tess-goddard-lcs/sector"+sectors[i]+"lookup.csv") ##Read the lookup table
    path = ref.query('TIC_ID == '+tics[i])["Filename"].iat[0] ##Locate the filepath to the light curve
    lc = load_lc(mountDir +"tess-goddard-lcs/"+path, mask = True)##Load the light curve

    lc = lc.flatten() ##Remove (or attempt to remove) the background variability
    lc = lc.normalize() ##Normalize the lightcurve
    lc = lc.remove_nans() ##Remove nans from data
    lc = lc.remove_nans('flux_err')
    lc = fold(lc) ##Phase fold the light curve, only do this if you know the light curve has is periodic within the available data
    times = lc.time.jd ##Load values into their own arrays
    flux = lc.flux.value
    fluxErr = lc.flux_err.value
    #times, flux, fluxErr = cutLightCurve(times, flux, fluxErr, lowerTimes[i], upperTimes[i]) ##Isolate the occultation event, don't have to do this if phase folding
    
    minFlux = np.min(flux) ##Identify the minimum value flux
    index = np.where(flux == minFlux)
    tRef = times[index[0][0]] ##Use the minimum flux timepoint as the initial guess for the center of the transit
    predictedRadius = 50*np.sqrt(1-minFlux) ##Predict the radius assuming the occulting object is a fully opaque circle
    #print(tRef)



    ncpu = cpu_count()
    print("{0} CPUs".format(ncpu))




    #Initial guess assumes radius based of opaque circle, transit lenght 2 days, the object is fully opaque, eclipses the equator of the star and is centered at the minimum flux value
    ##The guess are then spread by with a random distribution of the following widths
    ##Dimensions 5px, Period/Speed (1 stellar radii/day) 0.5 days -  4days, Opacity 0.8-1, Impact Parameter 0-5px, Transit Center 2 days
    ##The dimensions and impact parameter are based on a 50px stellar radius, if the star gets scaled, these also get scaled.
    pos = [predictedRadius,predictedRadius,1,0.9,0,tRef] * np.ones([16,6]) + [50,50,10,2,50,20]*((np.random.random([16,6])-0.5)/5)
    nwalkers, ndim = pos.shape

    ##Sample!!!!
    startTime = time.time()
    print("Beginning Sampling")
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args = (times, flux, fluxErr,minFlux), pool = pool, moves = emcee.moves.StretchMove(a = 2))
        sampler.run_mcmc(pos,20000)

    endTime = time.time()
    print(endTime-startTime)


    ##This block plots the traces of the 5parameters
    fig, axes = plt.subplots(7,figsize = (10,10), sharex = True)
    samples = sampler.get_chain()
    logProb = sampler.get_log_prob()
    labels = ["X dim", "Y dim", "Pixel Speed", "Opacity", "Impact Parameter", "Reference Time"]
    for i in range(ndim):
        axes[i].plot(samples[:,:,i],'k',alpha = 0.3)
        axes[i].set_xlim(0,len(samples))
        axes[i].set_ylabel(labels[i])
        
    axes[-1].plot(logProb[:,:],'k',alpha = 0.3)
    axes[-1].set_xlim(0,len(samples))
    axes[-1].set_ylabel("Log Probability")
    axes[-1].set_xlabel("Step Number")
    plt.savefig("imgs/tracesTIC"+tic+".png")

    fig = plt.subplot()




    ##This section takes 100 of the samples from the later 80% and plots then over the true curve
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        try:
            sample = flat_samples[ind]
            sampleFlux = transitSim(sample[0],sample[1],50,sample[2],times,sample[5],sample[3],int(sample[4]))
            fig.plot(times,sampleFlux,alpha = 0.1, color = 'orange')
        except:
            print("Invalid parameters", sample)
        

    fig.plot(times,flux, alpha = 0.5, color = 'blue',ls = '-', marker = 'o')
    plt.savefig("imgs/transitsTIC"+tic+".png")




    ##Make a corner plot
    flatSamples = sampler.get_chain(discard = 500, flat = True)
    fig = corner.corner(flatSamples, bins = 40, labels = labels)
    plt.savefig("imgs/cornerTIC"+tic+".png")

    for i in range(ndim):
        print(labels[i], np.mean(samples[:,:,i]),np.std(samples[:,:,i]))

    print("Acceptance Fraction: ",sampler.acceptance_fraction)