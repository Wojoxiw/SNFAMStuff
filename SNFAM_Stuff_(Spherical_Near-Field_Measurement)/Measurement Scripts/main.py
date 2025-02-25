#Script to use basement antenna lab measurement chamber with R&S ZVL13 VNA, using RPi connected by a coax to the old VNA's trigger signal to detect angles to make measurements at


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

# Tuning parameters
VNA_DELAY = .03 # in seconds, should be the time it takes for the VNA sweep to complete (or larger, but less than the wait time between sweeps)

# Pin config
TRIGGER_GPIO = 36


if __name__ == '__main__':
    # init switch GPIO
    #GPIO.setmode(GPIO.BCM)
    GPIO.setmode(GPIO.BOARD) ##to use 'physical board pin numbers', whatever that means... seems to be 1-2, 3-4 based on row
    
    #GPIO.setup(TRIGGER_GPIO, GPIO.IN,pull_up_down=GPIO.PUD_UP)
    #while(True):
    #    sleep(0.1)
    #    print(GPIO.input(TRIGGER_GPIO))
    
    # trigger callback function
    flag = False
    def button_pressed_callback(channel): ## this happens when the trigger event is detected
        #print('button_pressed_callback')
        global flag
        flag = True

    # init trigger GPIO
    GPIO.setup(TRIGGER_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(TRIGGER_GPIO, GPIO.RISING, 
            callback=button_pressed_callback, bouncetime=10) #bouncetime in ms, one way to try to prevent extra detections...
    
    # init VNA connection
    ZVL = VNA('169.254.178.48')
    ZVL.setFreqs()


    # init output file
    probePol = 0 ## 0 for horiz (co-pol at (0,0)), 90 if vert. With current setup, this requires manually mounting the probe at a 90 degree angle
    debugPrinting = True ## print stuff to see whats happening. Hopefully this doesn't lag the script
    path = '9.35GHzArray+uncloaked.csv'
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
    csvfile.write('Data for 9.35 GHz array, recorded on '+date.today().strftime('%d/%m/%Y')+'\n') ## date can be wrong because rpi does not update it
    csvfile.write('Measurement settings: Meas. BW: 100Hz, 4x Averaging, 0dBm power '+'\n')
    thetaStart = -100 ## in case multiple measurements need to be combined
    thetaStep = 1 ## all this in degrees
    thetaMax = 100
    nTheta = (thetaMax-thetaStart)/thetaStep+1 ## number of azimuthal angle measurements - typically from -100 to 100 with step of 1
    phiStep = 1 # starting angle. Must adjust this for each measurement
    phiStart = 0
    phiMax = 240 ## max phi angle, used to help prevent false accepted triggers
    nPhi = (phiMax-phiStart)/phiStep+1 ## number of roll angle measurements - typically from 0 to 240  Go to 240 instead of 180 since this positioner can be jittery at start/end at strange position. So take some intermediate 180 degrees
    csvfile.write('Format: # of triggers, time since start, probe pol, theta angle [degrees], phi angle, f1 [Hz], S21(f1), f2, S21(f2)\n')
    
    # init variables 
    trig_count = 0
    atc = 0 ## triggers accepted as being a measurement point
    rotatorSpeed = 4 ## deg/sec. ## start missing good triggers if this is too fast
    exp_meas_time_wait = phiStep/rotatorSpeed ## expected time between accepted triggers, in s
    meas_time_delt = .08 ## variation in the time - anything within this delta from the expected time is accepted as a measurement point. Still need to be careful, since one of the other random-ish times might coincide...
    print('Accepting times within '+str(meas_time_delt)+'s of '+str(exp_meas_time_wait)+'s')
    accepted = False
    c=0 ## count of seeming duplicate measurements
    
    magOld = 0
    t1 = time.time() ## initial time
    dtOld = t1 ## time of previous trigger detection
    going = True
    waiting = False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    
    phi=0
    theta=0
    
    def getThetaPhi(atc): ## returns the current theta, phi, calculated based on the accepted trigger count (atc)
        ### starting with phi
        theta = np.floor((atc-1)/nPhi)*thetaStep+thetaStart
        phi = phiStep*((atc-1)%nPhi)+phiStart
        
        ### starting with theta
        #theta = ((atc-1)%nTheta)*thetaStep+thetaStart
        #phi = np.floor(((atc-1)/nTheta))*phiStep+phiStart
        return theta, phi
    
    print('Waiting for signal:') ## Once this prints, run the measurement
    while going:
        if flag:
            flag = False
            dt = time.time() ##detection time
            delT = dt-dtOld
            #print() ## to make things more legible
            #print()
            dtOld = dt

            trig_count = trig_count + 1
            if(np.abs(delT-exp_meas_time_wait)<meas_time_delt): ## accept a trigger if the time since previous trigger indicates a measurement
                if(not accepted): ## if previous trigger not accepted, that data is also valid - to be saved
                    atc+=1 ## to count the previous, since it is accepted as a measurement
                    theta, phi = getThetaPhi(atc)
                    
                    datatofile = [trig_count, dt-t1, probePol, theta, phi] + data
                    csvwriter.writerow(datatofile)
                accepted = True
                atc+=1
            else:
                accepted = False
                
            if(debugPrinting):
                print("Time difference: {}".format(delT)+", Trigger/accepted count: {}".format(trig_count)+"/{}".format(atc)+', Accepted = '+str(accepted))
            
            ### The RPi seems to count one trigger per meas point
            ### starts with 1 or 2? (varies, hopefully based on settings and starting positioner position) triggers while moving to meas. start position
            ### 1 (or 2?) additional triggers when switching from azimuth sweep to roll sweep, + 1 trigger after finishing (when moving back to default position)
            
            # There should be #pts_azimuth * #pts_roll total accepted triggers
            
            # Sleep during sweep time (VNA is externally triggered during the intentionally accepted triggers, though using free run also? so just don't wait so long the next trigger happens)
            time.sleep(VNA_DELAY)
            #t = time.time()
            data = ZVL.meas()
            ## rotate phi, then theta, since the phi rotator can stutter a bit. Not sure which is better
            
            if(accepted):
                theta, phi = getThetaPhi(atc)
                
                mag = 20*np.log10(np.abs(data[3]))
                if(magOld==mag):
                    c+=1
                    print(str(c)+' duplicate measurements detected: VNA sweep time too long?') ## hopefully this will not go off unless there is actually a problem
                    if(c==5):
                        exit()
                magOld=mag
                if(debugPrinting):
                    print('Current Magnitude: '+str(mag)+', angle: ('+str(theta)+','+str(phi)+')')
                datatofile = [trig_count, dt-t1, probePol, theta, phi] + data
                csvwriter.writerow(datatofile)
                if(phi==phiMax): #theta==thetaMax ## whatever is being sweeped first, theta or phi
                    waiting = True
            if(waiting): ## after the sweep, wait for some seconds since it seems there can suddenly be a burst of false triggers here.
                tW = 13 ## wait time, adjust to be slightly less than it takes for the next sweep to start
                print('Sweep finished, waiting '+str(tW)+' s')
                time.sleep(tW)
                waiting = False
                
        if(atc > 0 and (time.time()-dt)>90):
            print('90 s since last accepted trigger - closing')
            going = False
            print('Expected trigger count:'+str(nTheta*nPhi)+', delta: '+str(nTheta*nPhi - atc)) ## if this doesn't equal the accepted trigger count, fix it and run again
            
    print('Total time taken: '+str(time.time()-t1)+'s')
    csvfile.close() ## needed to close the csvfile, otherwise nothing is saved, but only sometimes
    GPIO.cleanup()
    ZVL.close()
    exit()