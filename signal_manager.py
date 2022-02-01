from scipy.io import loadmat as load
import numpy as np
from scipy import signal

class BCI_signal():
  def __init__(self, dataset):
    self.data = load(dataset)# Conjunto de datos
    if ('C1' in self.data) and ('C2' in self.data):
      self.C1=np.array(self.data['C1'])  # Clase 1
      self.C2=np.array(self.data['C2'])  # Clase 2
    if 'signal' in self.data:            # Señal completa en caso de existir
      self.Signal=np.array(self.data['signal'])
      if (len(self.Signal)<len(self.Signal[0])):
        self.Signal=self.Signal.transpose()
      self.sig_len=len(self.Signal)       # Tamaño de la señal completa
    if 'samplingFreq' in self.data:       # Frecuencia de muestreo
      self.fs=self.data['samplingFreq'][0][0]
    else:                                # Si no está incluida en los datos, se ingresa por teclado
      self.fs=250
    
    self.window=self.fs*2                # Tamaño de la ventana
    self.noverlap=round(self.fs*1.5)     # Solapamiento de las ventanas

    self.channels=len(self.C1[:,0,0])   # Numero de canales existentes
    self.samples=len(self.C1[0,:,0])    # Número de muestras tomadas en cada experimento

    ###################################### Numero de muestras en cada clase
    if len(self.C1[0,0,:])==len(self.C2[0,0,:]):
      self.experiments=len(self.C1[0,0,:])  # Number of experiments in the dataset
    else:
      temp1 = len(self.C1[0,0,:])
      temp2 = len(self.C2[0,0,:])
      temp3 = np.arange(abs(temp1-temp2))
      if temp1 > temp2:
        self.C1 = np.delete(self.C1,temp3,axis=2)
      else:
        self.C2 = np.delete(self.C2,temp3,axis=2)
      self.experiments=len(self.C1[0,0,:])

      self.shape = np.shape(self.C1)
  
  def highPass_Filter(self,cutoff_frequency,filter_order,):
    fc = cutoff_frequency
    Wn = fc/(self.fs/2)
    num, den = signal.butter(filter_order,Wn,'highpass')
    try:
      self.Signal = signal.filtfilt(num, den, self.Signal,0)
    except AttributeError:
      pass
    for i in range(self.experiments):
      self.C1[:,:,i]=signal.filtfilt(num,den,self.C1[:,:,i],1)
      self.C2[:,:,i]=signal.filtfilt(num,den,self.C2[:,:,i],1)

  def lowPass_Filter(self,cutoff_frequency,filter_order,):
    fc = cutoff_frequency
    Wn = fc/(self.fs/2)
    num, den = signal.butter(filter_order,Wn,'lowpass')
    try:
      self.Signal = signal.filtfilt(num, den, self.Signal,0)
    except AttributeError:
      pass
    for i in range(self.experiments):
      self.C1[:,:,i]=signal.filtfilt(num,den,self.C1[:,:,i],1)
      self.C2[:,:,i]=signal.filtfilt(num,den,self.C2[:,:,i],1)
        
  def select_channels(self,channels_selected=None):
    size = len(channels_selected)
    if (channels_selected == None) or (size>self.channels):
      pass
    else:
      temp1=np.zeros((size,self.samples,self.experiments))
      temp2=np.zeros((size,self.samples,self.experiments))
      for i in range (size):
        temp1[i,:,:] = self.C1[channels_selected[i],:,:]
        temp2[i,:,:] = self.C2[channels_selected[i],:,:]
      self.C1 = temp1
      self.C2 = temp2
      self.channels = size