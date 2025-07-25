# INSPEX
A GUI for In-Situ electron energy spectral analysis

Intended as an in-situ equivalent to OSPEX for HXRs, this software provides a user-friendly pipeline to download, analyse and fit in-situ electron spectra. The guiding methodologyis laid out in Carter et al 2025 (in prep).

This software was written using Anaconda, and I reccomend that distribution is used to run the software. In addition to the packages included in that ditribution, the following packages are also required:
lmfit
solo_epd_loader
pickle

INSPEX can be used in two ways: by calling the script after the user has generated their own spectrum, and by using the GUI to complete the analysis from the start. 

Data Loading
To begin the GUI method, call the function instrument_choice at the console after running the INSPEX script. This will open a GUI window, shown in figure !!, with options for loading data. Here, the user has the option to load data using INSPEX's built in loading tools, which currently includes SolO STEP and STEREO STE-D. There is also an option to load spectral data from a user defined file, skipping the generation stage and going straight to the fitting stage. Before proceeding to the next stage, the user must also select a method of spectrum generation from instantaneous, peak flux, or fluence. This will change the following window's functions.

Spectral Generation
Once the time series data and spectral type are selected, we generate the spectrum. The user can use sliders or entry fields to select the background range, and integration time or instantaneous points depending on the method selected. For the instantaneous method, the user can select a number of different points and fit each in turn. These can be manually selected or generated at a regular time interval.

Fitting GUI
Once the spectrum is generated, the fitting GUI window is created, as shown in figure !!. In addition to the main window, a secondary window opens to show the user the spectrum. To begin defining the function to be fitted the user may select function components from the drop-down menu, or load a previously defined function from a file. The user can limit the fit to certain energy ranges using the entry fields beneath the function parameters. Each parameter takes an initial value, and an upper and lower bound. Fitting works best when the initial guess is good and the parameters are well constrained. Parameters can also be fixed by unchecking the vary box next to each. To get an idea of the initial guess, the preview parameters button generates a plot of the function with the currently defined initial guess values. The spectrum can also be saved at this stage to save time during later fitting attempts. The perform fit button runs the minimiser, and the results are displayed with the residuals in a separate window. Fitted parameter uncertainties are displayed at the console, though will be available in the GUI in later versions of this software. Once fitting is complete, the user can adjust parameters and try again, or choose to save their fitted parameters to a txt file, from which they can be loaded later. This txt file also contains the reduced chi squared and BIC values.

Via Script
To call INSPEX as a function to fit externally generated spectral data, it is simply called using the inspex.inspex function which takes the energies, spectral fluxes and flux uncertainties and opens the same fitting GUI as described in the Fitting GUI section.
