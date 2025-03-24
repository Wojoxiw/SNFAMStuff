"""Wrote to work with the R&S ZVL13, run from a raspberry pi connected to it with an ethernet cable. Presumably works with whatever other VNA also"""
from RsInstrument import *
from time import sleep
import numpy as np

class VNA(): # takes the IP address of the VNA as a string - can check with ipconfig on the VNA
    def __init__(self, ip): # connects to the VNA, then sets stuff up
        resource = 'TCPIP0::'+ip+'::INSTR'  # VISA resource string for the device
        
        self.VNA = RsInstrument(resource, True, False, "SelectVisa='rs'") ## this stuff is straight out of an R&S example. Reset will actually reset all settings, I do not do that
        """
        (resource, True, True, "SelectVisa='rs'") has the following meaning:
        (VISA-resource, id_query, reset, options)
        - id_query: if True: the instrument's model name is verified against the models 
        supported by the driver and eventually throws an exception.   
        - reset: Resets the instrument (sends *RST) command and clears its status syb-system
        - option SelectVisa:
                    - 'SelectVisa = 'socket' - uses no VISA implementation for socket connections - you do not need any VISA-C installation
                    - 'SelectVisa = 'rs' - forces usage of Rohde&Schwarz Visa
                    - 'SelectVisa = 'ni' - forces usage of National Instruments Visa     
        """
        sleep(1)  # Eventually add some waiting time when reset is performed during initialization. This line is probably needed
        
        print(f'VISA Manufacturer: {self.VNA.visa_manufacturer}')  # Confirm VISA package to be chosen
        self.VNA.visa_timeout = 5000  # Timeout for VISA Read Operations
        self.VNA.opc_timeout = 5000  # Timeout for opc-synchronised operations
        self.VNA.instrument_status_checking = True  # Error check after each command, can be True or False
        self.VNA.clear_status()  # Clear status register
        print('VNA Initialization Complete')
        
    def setFreqs(self, freqs = ['9.25', '9.35'], meas = 'S21'): # to measure 3 points in a freq range, given in GHz
        points = len(freqs)
        self.VNA.write_str('SENSe1:FREQuency:STARt '+str(freqs[0])+'GHZ')  # Start frequency to whatever
        self.VNA.write_str('SENSe1:FREQuency:STOP '+str(freqs[len(freqs)-1])+'GHZ')  # Stop frequency to whatever
        self.VNA.write('SENSe1:SWEep:POINts ' + str(points))  # Set number of sweep points to the defined number
        self.VNA.write_str('CALCulate1:PARameter:MEASure "Trc1", "'+meas+'"')  # Measurement now is S21
        
    def meas(self, b=1 , avg=0): ## take a single measurement, with specified averaging (not implemented, just choose on the VNA)
        #self.VNA.write_str('CALC:MARK ON') # turn on marker 1
        #self.VNA.write_str(':CALCulate1:MARKer1:FORMat COMPlex')  # set format to Re+jIm
        #data = self.VNA.query_str('CALCulate1:TRACe:MARKer1:YPOSition?')
        
        ## 100 Hz Meas. BW, 2 points, takes about .07s per sweep => approx. .28s for 4x averaging
        
        Ss = np.fromstring(self.VNA.query_str(':CALCULATE1:DATA? SDATA'), sep=',')  # S data
        fs = np.fromstring(self.VNA.query_str(':CALCULATE1:DATA:STIMULUS?'), sep=',') # f data
        data = []
        b=0
        for i in range(len(Ss)):
            if(i%2==0): #data comes in Re1, Im1,... Ren, Imn. this makes complex numbers instead, following each frequency
                data.append(fs[b])
                b+=1
                data.append(Ss[i]+1j*Ss[i+1])
        return data
    
    def measS11(self): ## measures S11
        
        Ss = np.fromstring(self.VNA.query_str(':CALCULATE1:DATA? SDATA'), sep=',')  # S data / in Real, then Imag format
        fs = np.fromstring(self.VNA.query_str(':CALCULATE1:DATA:STIMULUS?'), sep=',') # f data
        return Ss, fs
    
    def close(self): ## when finished.
        self.VNA.close()
    
#some other commands
#self.VNA.query_opc() ## waits until the VNA has finished all queued tasks
    
#####################################################################
### Run if Main
#####################################################################
if __name__ == "__main__":
    # this won't be run when imported
    print('Connecting to VNA...')
    ZVL = VNA('169.254.178.48')
    ZVL.setFreqs()
    data = ZVL.meas()
    print(data)
    ZVL.close()
