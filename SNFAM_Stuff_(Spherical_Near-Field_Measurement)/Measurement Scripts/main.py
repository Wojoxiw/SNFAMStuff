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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from multiprocessing import Process, shared_memory, Queue

# Tuning parameters
VNA_DELAY = .03 # in seconds, should be the time it takes for the VNA sweep to complete (or larger, but less than the wait time between sweeps) Kind of deprecated, so I just keep it small

# Pin config
TRIGGER_GPIO = 36

logFile = open('consoleOutput.txt', 'w')
def printing(text): # print but also save to a log file
    print(text)
    if logFile:
        logFile.write(str(text)+'\n')

def livePlotting(fname, memshareName, arrShape, queue, extents): # try to use a memory-shared array to make live plots on a separate process. Update only every sweep to avoid lag
    sharedMemory = shared_memory.SharedMemory(name=memshareName)
    sharedArray = np.ndarray(arrShape, dtype=np.float32, buffer=sharedMemory.buf)
    plt.ion()
    fig, ax = plt.subplots(figsize = (7,7))
    im = ax.imshow(sharedArray.copy(), extent=extents, vmin=-95, vmax=-40)
    fig.colorbar(im)
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Phi (degrees)')
    plt.title(fname)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5) # updates the plot
    
    while True: # continuous loop
        if(queue.empty()):
            plt.pause(2)
        else:
            queueItem = queue.get()
            if(queueItem == 1):
                sleep(3) ## to... give it time to update?
                im.set_data(sharedArray.copy())
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1) # updates the plot
            elif(queueItem == None):
                break
    plt.ioff()
    plt.savefig(fname)
    plt.close(fig)

if __name__ == '__main__':
    # init switch GPIO
    #GPIO.setmode(GPIO.BCM)
    GPIO.setmode(GPIO.BOARD) ##to use 'physical board pin numbers', whatever that means... seems to be 1-2, 3-4 based on row
    
    #GPIO.setup(TRIGGER_GPIO, GPIO.IN,pull_up_down=GPIO.PUD_UP)
    #while(True):
    #    sleep(0.1)
    #    printing(GPIO.input(TRIGGER_GPIO))
    
    # trigger callback function
    flag = False
    def button_pressed_callback(channel): ## this happens when the trigger event is detected
        #printing('button_pressed_callback')
        global flag
        flag = True

    # init trigger GPIO
    GPIO.setup(TRIGGER_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(TRIGGER_GPIO, GPIO.RISING, 
            callback=button_pressed_callback, bouncetime=10) #bouncetime in ms, one way to try to prevent extra detections...
    
    # init VNA connection
    ZVL = VNA('169.254.178.48') ## this is the VNA's IP - check this is the same as the VNA's if you can't connect, and try restarting this device
    ZVL.setFreqs(freqs = ['9.25', '9.35']) ## freqs in GHz, ['1.045', '1.085'] or ['9.25', '9.35']

    makePlots = True ## if True, will try to make an updating plot as the data is collected. Current setup on rpi seems too slow for this + maybe it should run on another thread?
    # init output file
    scanMode = 'continuous'  ## the scan mode as in MIDAS... only continuous has been implemented, since step mode is inconsistent
    probePol = 90 ## 0 for horiz (co-pol at (0,0) with slot array), 90 if vert. With current setup, this requires manually mounting the probe at a 90 degree angle
    debugprinting = True ## print stuff to see whats happening. Hopefully this doesn't lag the script
    path = '9.35GHzSlotBareYetAgain'+'140+Pol'+str(probePol)+'.csv'
    if(os.path.isfile(path)): ## path already exists, ask whether to overwrite it
        test = input('CSV file already exists. Overwrite? (y/n)')
        if(test=='y'):
            printing('Overwriting.')
            os.remove(path)
        elif(test=='n'):
            printing('Not overwriting. Closing.')
            exit()
        else:
            printing('??')
            exit()
    csvfile = open(path,'a') ## need 2 measurements per AUT, 1 *2 for two polarizations
    csvwriter = csv.writer(csvfile)
    csvfile.write('Data for 9.35 GHz array, recorded on '+date.today().strftime('%d/%m/%Y')+'\n') ## date can be wrong because rpi does not update it
    csvfile.write('Measurement settings: Meas. BW: 100Hz, 4x Averaging, 0dBm power '+'\n')
    thetaStart = -100 ## in case multiple measurements need to be combined
    thetaStep = 0.8 ## all this in degrees
    thetaStop = 100
    nTheta = int((thetaStop-thetaStart)/thetaStep)+1 ## number of azimuthal angle measurements - typically from -100 to 100 with step of 1
    thetaArray = np.linspace(thetaStart, thetaStop, nTheta)
    phiStep = .8 
    phiStart = 140 # starting angle.
    phiStop = 240 ## max phi angle, used to help prevent false accepted triggers
    nPhi = int((phiStop-phiStart)/phiStep)+1 ## number of roll angle measurements - typically from 0 to 240  Go to 240 instead of 180 since this positioner can be jittery at start/end at strange position. So take some intermediate 180 degrees
    phiArray = np.linspace(phiStart, phiStop, nPhi)
    phiSweeping = False ## Sweeping phi for different values of theta (inner sweep is phi), if not then its theta (this is the inner sweep)
    csvfile.write('Format: # of triggers, time since start, probe pol, theta angle [degrees], phi angle, f1 [Hz], S21(f1), f2, S21(f2)\n')
    
    # init variables 
    trig_count = 0
    atc = 0 ## triggers accepted as being a measurement point
    rotatorSpeed = 1.4 ## deg/sec. ## start missing good triggers if this is too fast
    exp_meas_time_wait = np.abs(thetaStep)/rotatorSpeed ##phiStep/rotatorSpeed ## expected time between accepted triggers, in s
    meas_time_delt = .145 ## variation in the time - anything within this delta from the expected time is accepted as a measurement point. Still need to be careful, since one of the other random-ish times might coincide...
    printing('Accepting times within '+str(meas_time_delt)+'s of '+str(exp_meas_time_wait)+'s')
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
            printing('Nonexistant scan mode -- stopping.')
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
        csvwriter.writerow(datatofile) ## update the datafile with the datum
        if(makePlots): ## also update the plot
            i = np.argmin(np.abs(thetaArray - theta))
            j = np.argmin(np.abs(phiArray - phi))
            magArray[j, i] = mag.astype(np.float32)
            np.copyto(plotArray, magArray)
        
    ###### the running loop
    printing('Waiting for signal:') ## Once this prints, run the measurement
    if(makePlots): ## initialize the plot, and show it
        arrShape = (nPhi, nTheta)
        memshare = shared_memory.SharedMemory(create=True, size=np.prod(arrShape)*4)
        magArray = np.zeros(arrShape, dtype=np.float32) - 120 ## magnitude array.
        plotArray = np.ndarray(arrShape, dtype=np.float32, buffer=memshare.buf) # a separate one for the plotter to read, to avoid mem issues?
        np.copyto(plotArray, magArray)
        queue = Queue()
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        plotProcess = Process(target=livePlotting, args=(path+'.png', memshare.name, arrShape, queue, [thetaStart, thetaStop, phiStop, phiStart]))
        plotProcess.start()
        
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
                
            if(debugprinting):
                printing("Time difference: {}".format(delT)+", Trigger/accepted count: {}".format(trig_count)+"/{}".format(atc)+', Accepted = '+str(accepted))
            
            # Sleep during sweep time (free run VNA, so time this)
            time.sleep(VNA_DELAY)
            #t = time.time()
            data = ZVL.meas() ## query the VNA for the measurement
            mag = 20*np.log10(np.abs(data[3]))
            
            if(accepted):
                writeData()
                if(magOld==mag):
                    c+=1
                    printing(str(c)+' duplicate measurements detected: VNA sweep time too long?') ## hopefully this will not go off unless there is actually a problem
                    if(c==5):
                        exit()
                magOld=mag
                if(debugprinting):
                    printing('Current Magnitude: '+str(mag)+', angle: ('+str(theta)+','+str(phi)+')')
                if( (theta==thetaStop and not phiSweeping) or (phi==phiStop and phiSweeping)  ): ## whatever is being sweeped first, theta or phi
                    waiting = True
            if(waiting): ## after the sweep, wait for some seconds since it seems there can suddenly be a burst of false triggers here.
                tW = 13#30 ## wait time, adjust to be slightly less than it takes for the next sweep to start
                printing('Sweep finished, waiting '+str(tW)+' s')
                if(makePlots):
                    queue.put(1) ## update the plot
                time.sleep(tW)
                waiting = False
                
        if(atc > 0 and (time.time()-dt)>90):
            printing('90 s since last accepted trigger - closing')
            going = False
            printing('Expected trigger count:'+str(nTheta*nPhi)+', delta: '+str(nTheta*nPhi - atc)) ## if this doesn't equal the accepted trigger count, fix it and run again
            
    tTot = time.time()-t1
    printing('Total time taken: '+str(tTot)+'s ('+str(tTot/3600)+' hours)')
    csvfile.close() ## needed to close the csvfile, otherwise nothing is saved, but only sometimes
    GPIO.cleanup()
    ZVL.close()
    if(makePlots): ## save the plot after the measurement
        memshare.close()
        memshare.unlink()
        #input('Execution paused to show plot. Enter anything to proceed.') # This turns out to be more annoying than opening the saved plot
        queue.put(None) # ends the plot, saving it
        plotProcess.join()
    exit()