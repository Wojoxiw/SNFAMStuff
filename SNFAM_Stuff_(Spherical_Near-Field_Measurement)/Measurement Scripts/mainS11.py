#Script to use basement antenna lab measurement chamber with R&S ZVL13 VNA, using RPi to read S11 to file
import signal
import sys
import RPi.GPIO as GPIO
import time
from vna import VNA
import numpy as np
import csv
from time import sleep
from datetime import date
import os

if __name__ == '__main__':
    t1 = time.time() ## initial time
    # init VNA connection
    ZVL = VNA('169.254.178.48') ## this is the VNA's IP - check this is the same as the VNA's if you can't connect, and try restarting this device
    freqs = np.linspace(.5, 1.5, 3333)
    ZVL.setFreqs(freqs = freqs, meas = 'S11') ## freqs in GHz, ['1.045', '1.085'] or ['9.25', '9.35']

    path = 'S11forUncloakedAntenna.csv'
    if(os.path.isfile(path)): ## path already exists, ask whether to overwrite it
        test = input('CSV file already exists. Overwrite? (y/n)')
        if(test=='y'):
            print('Overwriting.')
            os.remove(path)
        elif(test=='n'):
            print('Not overwriting. Closing.')
            exit()
        else:
            print('??')
            exit()
    csvfile = open(path,'a') ## need 2 measurements per AUT, 1 *2 for two polarizations
    csvwriter = csv.writer(csvfile)
    csvfile.write('Data for 1.045 GHz dipole antenna, recorded on '+date.today().strftime('%d/%m/%Y')+'(date is not accurate)\n') ## date can be wrong because rpi does not update it
    csvfile.write('Measurement settings: Meas. BW: 1000Hz, 4x Averaging, 0dBm power '+'\n')
    csvfile.write('Format: frequency, 20*log10(|S11(frequency)|)\n')
    

    Ss, fs = ZVL.measS11() ## query the VNA for the measurement
    for i in range(len(fs)): # go through each row
        csvwriter.writerow((fs[i], 20*np.log10(np.abs(Ss[2*i] + 1j*Ss[2*i+1]))))#csvwriter.writerow((fs[i], Ss[2*i], Ss[2*i+1]))
            
    print('Total time taken: '+str(time.time()-t1)+'s')
    csvfile.close() ## needed to close the csvfile, otherwise nothing is saved, but only sometimes
    ZVL.close()
    exit()