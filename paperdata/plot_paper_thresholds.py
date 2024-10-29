import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def fit_function(x_data, *params):
    """ Fitting function. """

    d, p = x_data
    p_th, nu, A, B, C = params
    x = (p - p_th)*d**nu
    
    return A*x**2 + B*x + C

def get_fit_params(xdata, ydata, params_0=None, ftol: float = 1e-5, maxfev: int = 2000):
    """Get fitting params."""

    # Curve fit.
    bounds = [min(xdata[1]), max(xdata[1])]
    if params_0 is not None and params_0[0] not in bounds:
        params_0[0] = (bounds[0] + bounds[1]) / 2
    params_opt, _ = curve_fit(fit_function, xdata, ydata, p0=params_0, ftol=ftol, maxfev=maxfev)

    # print(pcov[0,0]**0.5)

    return params_opt
    
def get_fit(d, p_list, params):
    """Fitting function."""

    p_th, nu, A, B, C = params
    return [A*((p - p_th)*d**nu)**2 + B*((p - p_th)*d**nu) + C for p in p_list]

def rescale_ps(d, ps, params):
    """Rescaled error rate using threshold fit."""
    p_th, nu, A, B, C = params
    return [(p - p_th) * d**(1/nu) for p in ps]

def linear_fit(x, a, b):
    return a*x + b

# Plotting 

opacity = 1.0

# TORIC RESTRICTED 

# Odd toric code
toric_data = pd.read_csv('./data/toric-mc/toric_restricted.csv')

toric_data = toric_data.loc[toric_data['p']>0.151]
toric_data = toric_data.loc[toric_data['p']<0.158]
sizes = [12, 20, 40, 60] #  [10, 22, 42, 62] 
#color = cm.rainbow(np.linspace(0.0, 0.4, len(sizes)))

xdata = toric_data[['L', 'p']].T.values
ydata = toric_data[['p_log']].values.flatten()
fit_data = get_fit_params(xdata, ydata, params_0=[0.15,0.61,0.5,0.5,0.15])

print('Toric fits restricted:', fit_data)

color = cm.rainbow(np.linspace(0.0, 0.4, len(sizes)))

fig, ax = plt.subplots()

for d, c in zip(sizes, color): 

    dfd = toric_data.loc[toric_data['L']==d]
    ax.plot(dfd['p'], get_fit(int(d), dfd['p'], fit_data), '--', c=c, label=f'd = {int(d/2)} res.', dashes=(5, 3), linewidth=1.1, alpha=opacity)
    errs = np.sqrt( dfd['p_log']*(1-dfd['p_log']) / dfd['n_runs'] )
    plt.errorbar(dfd['p'], dfd['p_log'], yerr=errs, marker='.', c=c, ls='None',  markersize=5, capsize=2, linewidth=1.3, alpha=opacity)


ax.axvline(x=fit_data[0], linewidth=1, color='lightgrey', label=r'res. $p_{th} = $'+f'{round(fit_data[0]*100, 2)} %')
ax.legend( loc='lower right')
ax.set_xlabel(r"$p$", fontsize = 12)
ax.set_ylabel(r"$P_{fail}$", fontsize = 14)


# # TORIC UNIFIED 

toric_data_uni_df = pd.read_csv('./data/toric-mc/toric_unified_config_c_data.csv')

th_fits = []

ax2 = ax.twinx()
ax.set_yscale('log')
ax2.set_yscale('log')
# ax.set_ylim(0.094, 0.178)
# ax2.set_ylim(0.094, 0.178)

# for w in np.unique(toric_data_uni_df['weight']):
for w in [0.5]: 
    print(w)

    toric_data_uni = toric_data_uni_df.loc[toric_data_uni_df['weight']==w]
    # sizes = [12, 20, 40, 60] # [10, 22, 42, 62] 
    color = cm.rainbow(np.linspace(0.6, 1.0, len(sizes)))

    # # Plot threshold 

    # toric_data_uni = toric_data_uni.loc[toric_data_uni['p']>0.147]
    # toric_data_uni = toric_data_uni.loc[toric_data_uni['p']<0.156]

    # toric_data_uni = toric_data_uni.loc[toric_data_uni['p']>0.10]
    # toric_data_uni = toric_data_uni.loc[toric_data_uni['p']<0.156]

    xdata = toric_data_uni[['L', 'p']].T.values
    ydata = toric_data_uni[['p_log']].values.flatten()
    fit_data = get_fit_params(xdata, ydata, params_0=[0.1,0.7,0.3,0.4,0.1])

    print('Toric fits unified:', fit_data)

    th_fits.append(fit_data[0])
    #color = cm.rainbow(np.linspace(0.0, 1.0, len(sizes)))

    lines = []

    for d, c in zip(sizes, color):

        dfd = toric_data_uni.loc[toric_data_uni['L']==d]
        ax2.plot(dfd['p'], get_fit(int(d), dfd['p'], fit_data), '--', c=c, label=f'd = {int(d/2)} uni.', dashes=(5, 3), linewidth=1.1, alpha=opacity)
        errs = np.sqrt( dfd['p_log']*(1-dfd['p_log']) / dfd['n_runs'] )
        ax2.errorbar(dfd['p'], dfd['p_log'], yerr=errs, marker='.', c=c, ls='None',  markersize=5, capsize=2, linewidth=1.3, alpha=opacity)

    ax2.axvline(x=fit_data[0], linewidth=1, color='darkgrey', label=r'uni. $p_{th} = $'+f'{round(fit_data[0]*100, 2)} %')
    ax2.legend(loc="upper left")
    ax2.get_yaxis().set_visible(False)
    plt.savefig(f'paperfigs/toric_code_unified_even_distance_configC_w={w}.png', bbox_inches='tight')
    plt.show() 


##############################################################################################################################################
################################################ COLOR CODE PLOTS ############################################################################


color_data = pd.read_csv('./data/color-mc/color_restricted.csv')

sizes = [10, 20, 30, 40] 

# Plot restricted threshold 
color_data = color_data.loc[color_data['p'] > 0.098]
color_data = color_data.loc[color_data['p'] < 0.104]

xdata = color_data[['L', 'p']].T.values
ydata = color_data[['p_log']].values.flatten()
fit_data = get_fit_params(xdata, ydata, params_0=[1,1,1,1,1])
color = cm.rainbow(np.linspace(0.0, 0.4, len(sizes)))

# Create figrue for color code 
# fig, ax = plt.subplots()


fig, (ax, ax2) = plt.subplots(nrows=2, sharex=False, sharey=False)
fig.set_size_inches(6, 6.5)

ax2.tick_params(labelbottom=True)

for d, c in zip(sizes, color): 

    dfd = color_data.loc[color_data['L']==d] 
    errs = dfd['p_log']*(1-dfd['p_log']) / np.sqrt(dfd['n_runs']) 
    errs = np.sqrt(dfd['p_log']*(1-dfd['p_log']) / dfd['n_runs'])
    ax.plot(dfd['p'], get_fit(int(d), dfd['p'], fit_data), '--', c=c, label=f'd = {int(d)} res.', dashes=(5, 3), linewidth=1.1, alpha=opacity)
    ax.errorbar(dfd['p'], dfd['p_log'], yerr=errs, marker='.', c=c, ls='None',  markersize=5, capsize=2, linewidth=1.3, alpha=opacity)

ax.axvline(x=fit_data[0], linewidth=1, color='darkgrey', label=r'res. $p_{th} = $'+f'{round(fit_data[0]*100, 2)} %')
ax.legend( loc='lower right')
# ax.set_xlabel(r"$p$", fontsize = 12)
ax.set_ylabel(r"$P_{fail}$", fontsize = 14)


print('Color restricted threshold fit parameters:', fit_data)

# COLOR UNIFIED : W = 2.1 

color_data_df = pd.read_csv('./data/color-mc/color_unified.csv')

th_fits = []

# ax2 = ax.twinx()
ax.set_yscale('log')
ax.set_ylim(0.13, 0.19)

# pick a weight 
w = 2.1
    
color_data_df = color_data_df.loc[color_data_df['p'] > 0.098]
color_data_df = color_data_df.loc[color_data_df['p'] < 0.104]
color_data = color_data_df.loc[color_data_df['weight']==w]
sizes = np.unique(color_data['L'])
color = cm.rainbow(np.linspace(0.6, 1.0, len(sizes)))

# Plot threshold 
xdata = color_data[['L', 'p']].T.values
ydata = color_data[['p_log']].values.flatten()
fit_data = get_fit_params(xdata, ydata, params_0=[1,1,1,1,1])
th_fits.append(fit_data[0])

lines = []

for d, c in zip(sizes, color):

    dfd = color_data.loc[color_data['L']==d]
    ax2.plot(dfd['p'], get_fit(int(d), dfd['p'], fit_data), '--', c=c, label=f'd = {int(d)} uni.', dashes=(5, 3), linewidth=1.1, alpha=opacity)
    errs = np.sqrt( dfd['p_log']*(1-dfd['p_log']) / dfd['n_runs'] )
    ax2.errorbar(dfd['p'], dfd['p_log'], yerr=errs, marker='.', c=c, ls='None',  markersize=5, capsize=2, linewidth=1.3, alpha=opacity)

ax2.axvline(x=fit_data[0], linewidth=1, color='darkgrey', label=r'uni. $p_{th} = $'+f'{round(fit_data[0]*100, 2)} %')
# ax2.legend(loc="upper left")
ax2.legend( loc='lower right')
ax2.get_yaxis().set_visible(True)
ax2.set_xlabel(r"$p$", fontsize = 12)
ax2.set_ylabel(r"$P_{fail}$", fontsize = 14)
ax2.set_yscale('log')
ax2.set_ylim(0.13, 0.19)

plt.savefig(f'paperfigs/color_code_unified_A_{w}_B_1+restricted.pdf', bbox_inches='tight')
plt.show()
