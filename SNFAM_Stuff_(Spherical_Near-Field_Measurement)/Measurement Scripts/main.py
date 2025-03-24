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
    ZVL = VNA('169.254.178.48') ## this is the VNA's IP - check this is the same as the VNA's if you can't connect, and try restarting this device
    ZVL.setFreqs(freqs = ['9.25', '9.35']) ## freqs in GHz, ['1.045', '1.085'] or ['9.25', '9.35']


    # init output file
    scanMode = 'continuous'  ## the scan mode as in MIDAS... only continuous since step mode is inconsistent
    probePol = 0 ## 0 for horiz (co-pol at (0,0) with slot array), 90 if vert. With current setup, this requires manually mounting the probe at a 90 degree angle
    debugPrinting = True ## print stuff to see whats happening. Hopefully this doesn't lag the script
    path = '9.35GHzSlotArray+ManyCloakedsFixedMotortheta-100to0phito240'+'Pol'+str(probePol)+'.csv'
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
    thetaStep = 0.8 ## all this in degrees
    thetaStop = 0
    nTheta = (thetaStop-thetaStart)/thetaStep+1 ## number of azimuthal angle measurements - typically from -100 to 100 with step of 1
    phiStep = .8 # starting angle. Must adjust this for each measurement
    phiStart = 0
    phiStop = 240 ## max phi angle, used to help prevent false accepted triggers
    nPhi = (phiStop-phiStart)/phiStep+1 ## number of roll angle measurements - typically from 0 to 240  Go to 240 instead of 180 since this positioner can be jittery at start/end at strange position. So take some intermediate 180 degrees
    phiSweeping = True ## phi, if not then its theta (this is the inner sweep)
    csvfile.write('Format: # of triggers, time since start, probe pol, theta angle [degrees], phi angle, f1 [Hz], S21(f1), f2, S21(f2)\n')
    
    # init variables 
    trig_count = 0
    atc = 0 ## triggers accepted as being a measurement point
    rotatorSpeed = 1.4 ## deg/sec. ## start missing good triggers if this is too fast
    exp_meas_time_wait = np.abs(thetaStep)/rotatorSpeed ##phiStep/rotatorSpeed ## expected time between accepted triggers, in s
    meas_time_delt = .145 ## variation in the time - anything within this delta from the expected time is accepted as a measurement point. Still need to be careful, since one of the other random-ish times might coincide...
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
        if(phiSweeping):
            theta = np.floor((atc-1)/nPhi)*thetaStep+thetaStart
            phi = phiStep*((atc-1)%nPhi)+phiStart
        elif(not phiSweeping):
            theta = ((atc-1)%nTheta)*thetaStep+thetaStart
            phi = np.floor(((atc-1)/nTheta))*phiStep+phiStart
        return theta, phi
    
    def acceptTrigger(scanMode, delT): ## decides whether to accept the trigger or not
        if(scanMode == 'continuous'):
            if(np.abs(delT-exp_meas_time_wait)<meas_time_delt): ## accept a trigger if the time since previous trigger indicates a measurement
                return True
            else:
                return False
        elif(scanMode == 'step'): ## step mode
            return False ## step mode gives inconsistent trigger timings - does not work
        else:
            print('Nonexistant scan mode -- stopping.')
            exit()
            
    def acceptedTrigger(): ## does the things that happen when a trigger is accepted as corresponding to a measurement point
        global accepted
        global atc
        if(not accepted and ((phiSweeping and getThetaPhi(atc)[1]==phiStop) or (not phiSweeping and getThetaPhi(atc)[0]==thetaStop))):
            ## if previous trigger not accepted, but it is the first trigger of a sweep, that data is also valid - to be saved. Sometimes during the sweep a .05s trigger comes out of nowhere - this excludes those
            atc+=1
            writeData()
        else:
            pass ## writeData occurs later in the loop
        accepted = True ## the previous trigger was accepted
        atc+=1
        
    def writeData(): ## after an accepted trigger, save the data
        global theta, phi
        global datatofile
        theta, phi = getThetaPhi(atc)
        datatofile = [trig_count, dt-t1, probePol, theta, phi] + data
        csvwriter.writerow(datatofile)
        
    ###### the running loop
    print('Waiting for signal:') ## Once this prints, run the measurement
    while going:
        if flag: ## trigger signal is active
            flag = False
            dt = time.time() ##detection time
            delT = dt-dtOld
            dtOld = dt

            trig_count = trig_count + 1
            if(acceptTrigger(scanMode, delT)): ## check if to accept it - if so, do stuff
                acceptedTrigger()
            else: ## if not, note that previous trigger was not accepted
                accepted = False
                
            if(debugPrinting):
                print("Time difference: {}".format(delT)+", Trigger/accepted count: {}".format(trig_count)+"/{}".format(atc)+', Accepted = '+str(accepted))
            
            # Sleep during sweep time (free run VNA, so time this)
            time.sleep(VNA_DELAY)
            #t = time.time()
            data = ZVL.meas() ## query the VNA for the measurement
            
            if(accepted):
                writeData()
                
                mag = 20*np.log10(np.abs(data[3]))
                if(magOld==mag):
                    c+=1
                    print(str(c)+' duplicate measurements detected: VNA sweep time too long?') ## hopefully this will not go off unless there is actually a problem
                    if(c==5):
                        exit()
                magOld=mag
                if(debugPrinting):
                    print('Current Magnitude: '+str(mag)+', angle: ('+str(theta)+','+str(phi)+')')
                if( (theta==thetaStop and not phiSweeping) or (phi==phiStop and phiSweeping)  ): ## whatever is being sweeped first, theta or phi
                    waiting = True
            if(waiting): ## after the sweep, wait for some seconds since it seems there can suddenly be a burst of false triggers here.
                tW = 13#30 ## wait time, adjust to be slightly less than it takes for the next sweep to start
                print('Sweep finished, waiting '+str(tW)+' s')
                time.sleep(tW)
                waiting = False
                
        if(atc > 0 and (time.time()-dt)>90):
            print('90 s since last accepted trigger - closing')
            going = False
            print('Expected trigger count:'+str(nTheta*nPhi)+', delta: '+str(nTheta*nPhi - atc)) ## if this doesn't equal the accepted trigger count, fix it and run again
            
    tTot = time.time()-t1
    print('Total time taken: '+str(tTot)+'s ('+str(tTot/3600)+' hours)')
    csvfile.close() ## needed to close the csvfile, otherwise nothing is saved, but only sometimes
    GPIO.cleanup()
    ZVL.close()
    exit()