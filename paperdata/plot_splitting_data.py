import math 
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

####### ANALYTICAL FITS ########

# COLOR CODE
def path_counting_restricted(M):
    return (M/2) * np.sum([math.comb(M, k) * math.comb(M - k, k) * 4**(M - k) for k in range(int(np.floor(M/2)) + 1)])

def path_counting_unified(M):
    return (M/2) * np.sum([math.comb(M, k) * math.comb(M - k, k) * 4**(M - (3*k/2)) for k in range(int(np.floor(M/2)) + 1)])

def analytical_ler(L, per, decoder):
    n = L
    if decoder=='restricted':
        return [path_counting_restricted(int(L/2)) * p**(L/2) * (1-p)**(n - L/2) for p in per] 
    if decoder=='unified':
        return [path_counting_unified(int(L/2)) * p**(L/2) * (1-p)**(n - L/2) for p in per] 
    

####### PLOT RESTRICTED COLOR CODE ##############

data = pd.read_csv('./data/color-splitting/color_restricted.csv')

sizes = [s for s in np.unique(data['L'])] #int(s/2) % 2 == 0] 
# color = cm.rainbow(np.linspace(0.0, 0.4, len(sizes)))
color = cm.rainbow(np.linspace(0.0, 1.0, len(sizes)))

for L, c in zip(sizes, color):
    
    df = data.loc[data['L']==L]
    if L == 6:
        # plt.plot(df['p'][:15], df['p_log'][:15], '.', c=c, label=f'$d = ${L} res.')
        plt.plot(df['p'][:15], df['p_log'][:15], 'o', markersize=2.0, c=c, label=f'$d = ${L} res.')
        plt.plot(df['p'][10:-13], analytical_ler(L, df['p'], 'restricted')[10:-13], '--', c=c, linewidth=0.9)
    else:
        plt.plot(df['p'][:21], df['p_log'][:21], 'o', markersize=2.0, c=c, label=f'$d = ${L} res.')
        plt.plot(df['p'][:-20], analytical_ler(L, df['p'], 'restricted')[:-20], '--', c=c, linewidth=0.9)

#plt.loglog()
# plt.xlim(10**-6,0.02)
#plt.gcf().set_facecolor('white')
#plt.legend(loc="upper left")
#plt.xlabel(r"$p$", fontsize = 12)
#plt.ylabel(r"$P_{fail}$", fontsize = 14)
#plt.show()

####### UNIFIED COLOR CODE ##############

data = pd.read_csv('./data/color-splitting/color_unified.csv')
data = data.loc[data['L'] < 18]
data = data.loc[data['L'] > 6]

sizes = np.unique(data['L'])
# color = cm.rainbow(np.linspace(0.6, 1.0, len(sizes)))

for L, c in zip(sizes, color):
    
    df = data.loc[data['L']==L]
    if L == 6:
        plt.plot(df['p'][:21], df['p_log'][:21], '>', markersize=3, c=c, label=f'$d = {L}$ uni.')
        plt.plot(df['p'][:-20], analytical_ler(L, df['p'], 'unified')[:-20], '-', c=c, linewidth=1)
    else:
        plt.plot(df['p'][:21], df['p_log'][:21], '>', markersize=3, c=c, label=f'$d = {L}$ uni.')
        plt.plot(df['p'][:-20], analytical_ler(L, df['p'], 'unified')[:-20], '-', c=c, linewidth=1)

plt.loglog()
plt.gcf().set_facecolor('white')
plt.legend(ncols=2, columnspacing=1, edgecolor='white')
plt.xlabel(r"$p$", fontsize = 12)
plt.ylabel(r"$P_{fail}$", fontsize = 14)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
plt.savefig('./paperfigs/color_unified+restricted_splitting.pdf', bbox_inches='tight')
plt.show()


####### ANALYTICAL FITS ########


# TORIC CODE
'''Analytical expressions.'''
def path_counting_even_restricted(d):
    return (d/2)*math.comb(d, int(d/2)) * 2**(d/2)

def path_counting_odd_restricted(d):
    return d*((d+1)/2) * math.comb(d, int((d+1)/2)) * 2**((d+1)/2)  # with extra (d+1)/2 term I still don't understand 
    # return d * math.comb(d, int((d+1)/2)) * 2**((d+1)/2) 

def path_counting_even_unified(d):
    return (d/2)*math.comb(d, int(d/2)) 

def path_counting_odd_unified(d):
    # return ((d+1) / 2) * ( (d*((d+3)/2) * math.comb(d, int((d+1)/2))) + 2*(d-1)*math.comb(d-2, int((d-3)/2)) )
    return (d*((d+3)/2)* math.comb(d, int((d+1)/2))) + 2*(d-1)*math.comb(d-2, int((d-3)/2)) + math.comb(d, int((d+1)/2))*((d-1)**2 / 2) + (d-1)*np.sum([math.comb(j, k)*math.comb(d-j, int(((d-1)/2)-k)) for j in range(1, int((d-1)/2)) for k in range(0, int(j-1))])

def analytical_ler(d, per, decoder):
    n = d
    if decoder=='restricted': 
        if d % 2 == 0:
            return [path_counting_even_restricted(d) * (p/3)**(d/2) * (1-p)**(n - d/2) for p in per]
        elif d % 2 == 1:
            return [path_counting_odd_restricted(d) * (p/3)**((d+1)/2) * (1-p)**(n - (d+1)/2) for p in per]

    if decoder=='unified': 
        if d % 2 == 0:
            return [path_counting_even_unified(d) * (p/3)**(d/2) * (1-p)**(n - d/2) for p in per]
        elif d % 2 == 1:
            return [path_counting_odd_unified(d) * (p/3)**((d+1)/2) * (1-p)**(n - (d+1)/2) for p in per]


####### PLOT TORIC CODE ##############

# EVEN SIZES

data = pd.read_csv('./data/toric-splitting/toric_restricted.csv')

sizes = [s for s in np.unique(data['L']) if int(s/2) % 2 == 0]
color = cm.rainbow(np.linspace(0.0, 0.9, len(sizes)))

for L, c in zip(sizes, color):
    
    df = data.loc[data['L']==L]
    if L == 20:
        plt.plot(df['p'][1:20], df['p_log'][1:20], '.', c=c, label=f'$d = ${int(L/2)} res.')
        plt.plot(df['p'][1:14], analytical_ler(int(L/2), df['p'], 'restricted')[1:14], '--', c=c, linewidth=0.9)
    elif L == 8:
        plt.plot(df['p'][2:19], df['p_log'][2:19], '.', c=c, label=f'$d = ${int(L/2)} res.')
        plt.plot(df['p'][2:17], analytical_ler(int(L/2), df['p'], 'restricted')[2:17], '--', c=c, linewidth=0.9)
    elif L == 12:
        plt.plot(df['p'][:18], df['p_log'][:18], '.', c=c, label=f'$d = ${int(L/2)} res.')
        plt.plot(df['p'][:15], analytical_ler(int(L/2), df['p'], 'restricted')[:15], '--', c=c, linewidth=0.9)
    else:
        plt.plot(df['p'][:17], df['p_log'][:17], '.', c=c, label=f'$d = ${int(L/2)} res.')
        plt.plot(df['p'][:14], analytical_ler(int(L/2), df['p'], 'restricted')[:14], '--', c=c, linewidth=0.9)


data = pd.read_csv('./data/toric-splitting/toric_unified.csv')

for L, c in zip(sizes, color):
    df = data.loc[data['L']==L]
    if L == 20:
        plt.plot(df['p'][:16], df['p_log'][:16], '>', c=c, label=f'$d = ${int(L/2)} uni.')
        plt.plot(df['p'][:12], analytical_ler(int(L/2), df['p'], 'unified')[:12], '-', c=c, linewidth=0.9)
    elif L == 16:
        plt.plot(df['p'][:18], df['p_log'][:18], '>', c=c, label=f'$d = ${int(L/2)} uni.')
        plt.plot(df['p'][:14], analytical_ler(int(L/2), df['p'], 'unified')[:14], '-', c=c, linewidth=0.9)
    elif L == 8:
        plt.plot(df['p'][2:18], df['p_log'][2:18], '>', c=c, label=f'$d = ${int(L/2)} uni.')
        plt.plot(df['p'][2:17], analytical_ler(int(L/2), df['p'], 'unified')[2:17], '-', c=c, linewidth=0.9)
    elif L == 12:
        plt.plot(df['p'][:18], df['p_log'][:18], '>', c=c, label=f'$d = ${int(L/2)} uni.')
        plt.plot(df['p'][:15], analytical_ler(int(L/2), df['p'], 'unified')[:15], '-', c=c, linewidth=0.9)
        
plt.loglog()
plt.gcf().set_facecolor('white')
plt.legend(ncols=2, columnspacing=1, edgecolor='white')
plt.xlabel(r"$p$", fontsize = 12)
plt.ylabel(r"$P_{fail}$", fontsize = 14)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
plt.savefig('./paperfigs/toric_splitting_restricted+unified_even_distance.pdf', bbox_inches='tight')
plt.show()


###### ODD SIZES TORIC CODE ##############

data = pd.read_csv('./data/toric-splitting/toric_restricted.csv')

sizes = [6, 10, 14, 18] # [s for s in np.unique(data['L']) if int(s/2) % 2 == 1]
color = cm.rainbow(np.linspace(0.0, 0.9, len(sizes)))

##### RESTRICTED DATA #######

for L, c in zip(sizes, color):
    
    df = data.loc[data['L']==L]
    if L == 18:
        plt.plot(df['p'][1:20], df['p_log'][1:20], '.', markersize=4, c=c, label=f'$d = ${int(L/2)} res.')
    elif L == 14:
        plt.plot(df['p'][2:19], df['p_log'][2:19], '.', markersize=4, c=c, label=f'$d = ${int(L/2)} res.')
    elif L == 10:
        plt.plot(df['p'][:18], df['p_log'][:18], '.', c=c, markersize=4, label=f'$d = ${int(L/2)} res.')
    elif L == 6:
        plt.plot(df['p'][:17], df['p_log'][:17], '.', c=c, markersize=4, label=f'$d = ${int(L/2)} res.')


##### UNIFIED DATA #######


data = pd.read_csv('./data/toric-splitting/toric_unified.csv')

for L, c in zip(sizes, color):

    df = data.loc[data['L']==L]
    if L == 18:
        plt.plot(df['p'][2:20], df['p_log'][2:20], '>', markersize=4, c=c, label=f'$d = ${int(L/2)} uni.')
    elif L == 14:
        plt.plot(df['p'][1:19], df['p_log'][1:19], '>', markersize=4, c=c, label=f'$d = ${int(L/2)} uni.')
    elif L == 10:
        plt.plot(df['p'][2:18], df['p_log'][2:18], '>', markersize=4, c=c, label=f'$d = ${int(L/2)} uni.')
    elif L == 6:
        plt.plot(df['p'][:18], df['p_log'][:18], '>', markersize=4, c=c, label=f'$d = ${int(L/2)} uni.')
        
plt.loglog()
plt.gcf().set_facecolor('white')
plt.legend(ncols=2, columnspacing=1, edgecolor='white')
plt.xlabel(r"$p$", fontsize = 12)
plt.ylabel(r"$P_{fail}$", fontsize = 14)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
plt.savefig('./paperfigs/toric_splitting_restricted+unified_odd_distance.pdf', bbox_inches='tight')
plt.show()