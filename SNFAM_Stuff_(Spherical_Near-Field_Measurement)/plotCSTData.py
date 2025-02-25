'''
Created on 11 okt. 2021

@author: al8032pa
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os

def plotLines():
    print('CST Plotting start.')
    clFN = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/secondfinal_try1_optimum.txt'

    unclFN = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/periodicuncloaked2.txt'
    
    unclData = np.transpose(np.loadtxt(unclFN, delimiter = ' ', skiprows = 630))
    #print(unclData)
    clData = np.transpose(np.loadtxt(clFN, delimiter = ' ', skiprows = 1675))
    
    plt.plot(clData[0], clData[1] + 10*np.log10(.005) - unclData[1] - 10*np.log10(.12), label = 'Second Final Design [CST]', linewidth = 2, color = 'tab:green', alpha = 0.8)
    print('CST Plotting Done.')
    
def plotLinesPaper2():
    print('CST Plotting start.')
    cl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Antenna.txt'
    uncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole.txt'
    
    unclData = np.transpose(np.loadtxt(uncl, delimiter = ',', skiprows = 3))
    clData = np.transpose(np.loadtxt(cl, delimiter = ',', skiprows = 3))
    
    plt.plot(clData[0], clData[1] - unclData[1], label = r'$\Delta \sigma_{\text{ecs}}$', linewidth = 2, color = 'tab:green', alpha = 0.8)
    plt.ylabel(r'$\sigma_{\text{ecs}}$ [dB]')
    plt.xlabel('Frequency [GHz]')
    plt.title('Cloaked Antenna Reduction in Extinction Cross-section')
    plt.legend()
    plt.tight_layout()
    
    slotBare = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array.txt'
    slotCl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+Cloakeds.txt'
    slotUncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+Uncloakeds.txt'
    
    slotBareData = np.transpose(np.loadtxt(slotBare, skiprows = 2))
    slotClData = np.transpose(np.loadtxt(slotCl, skiprows = 2))
    slotUnclData = np.transpose(np.loadtxt(slotUncl, skiprows = 2))
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(slotBareData[0], slotBareData[2], label = r'Bare Slot Array', linewidth = 1.5, linestyle = '-')
    ax2.plot(slotBareData[0], slotClData[2], label = r'with Cloaked', linewidth = 2, linestyle = ':')
    ax2.plot(slotBareData[0], slotUnclData[2], label = r'with Uncloaked', linewidth = 2, linestyle = '--', alpha = 0.8)
    plt.ylabel('Directivity [dBi]')
    plt.xlabel(r'$\theta$ [degrees]')
    plt.xlim(-85,85)
    plt.ylim(-29,38)
    plt.title('H-Plane Farfield Patterns')
    ax2.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    print('CST Plotting Done.')

if __name__ == '__main__':
    
    #plotlines()
    #plt.show()
    plotLinesPaper2()
    pass