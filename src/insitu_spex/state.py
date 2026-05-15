# state.py
# Shared GUI window state
fit_window = None
preview_window = None
resid_window = None

# Fit state
header = None
result = None
bic = None
redchi = None
init = None
minval = None
maxval = None
vary = None
parvals = None
param_uncert_calced = None
fit_summary = None
test_func = None
resids = None
fitmin=None
fitmax=None
entries = {}

# Model presence flags (which components are active)
therm_func_pres = None
bpl_pres = None
gauss_pres = None
power_pres = None
kappa_pres = None
bpl_and_therm_pres = None
double_therm_func_pres = None
tpl_pres = None
qpl_pres = None
quint_pl_pres = None