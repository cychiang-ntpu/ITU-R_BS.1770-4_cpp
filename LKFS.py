#!/usr/bin/env python
# coding: utf-8

# In[68]:


def calculate_loudness(signal, fs, G = [1.0, 1.0, 1.0, 1.41, 1.41]):
    # filter
    if len(signal.shape)==1: # if shape (N,), then make (N,1)
        signal_filtered = copy.copy(signal.reshape((signal.shape[0],1)))
    else:
        signal_filtered = copy.copy(signal)

    for i in range(signal_filtered.shape[1]):
        signal_filtered[:,i] = K_filter(signal_filtered[:,i], fs, False)

    # mean square
    T_g = 0.400 # 400 ms gating block
    Gamma_a = -70.0 # absolute threshold: -70 LKFS
    overlap = .75 # relative overlap (0.0-1.0)
    step = 1 - overlap

    T = signal_filtered.shape[0]/fs # length of measurement interval in seconds
    j_range = np.arange(0,int(np.round((T-T_g)/(T_g*step)))+1)
    z = np.ndarray(shape=(signal_filtered.shape[1],len(j_range)))

    # write in explicit for-loops for readability and translatability
    for i in range(signal_filtered.shape[1]): # for each channel i
        for j in j_range: # for each window j
            lbound = int(fs*T_g*j*step)
            hbound = int(fs*T_g*(j*step+1))
            z[i,j] = (1/(T_g*fs))*np.sum(np.square(signal_filtered[lbound:hbound,i]))

    G_current = np.array(G[:signal_filtered.shape[1]]) # discard weighting coefficients G_i unused channels
    n_channels = G_current.shape[0]

    l = [-.691 + 10.0*np.log10(np.sum([G_current[i]*z[i,j.astype(int)] for i in range(n_channels)])) \
             for j in j_range]

    # throw out anything below absolute threshold:
    indices_gated = [idx for idx,el in enumerate(l) if el > Gamma_a] 
    z_avg = [np.mean([z[i,j] for j in indices_gated]) for i in range(n_channels)]
    Gamma_r = -.691 + 10.0*np.log10(np.sum([G_current[i]*z_avg[i] for i in range(n_channels)])) - 10.0
    # throw out anything below relative threshold:
    indices_gated = [idx for idx,el in enumerate(l) if el > Gamma_r] 
    z_avg = [np.mean([z[i,j] for j in indices_gated]) for i in range(n_channels)]
    L_KG = -.691 + 10.0*np.log10(np.sum([G_current[i]*z_avg[i] for i in range(n_channels)]))

    return L_KG


# In[2]:


def print_coeff(coeff, filtname):
    print("<{}>".format(filtname))
    print("b_0:%20.15f"%coeff[0], " | ", "a_0%20.15f"%coeff[3])
    print("b_1:%20.15f"%coeff[1], " | ", "a_1%20.15f"%coeff[4])
    print("b_2:%20.15f"%coeff[2], " | ", "a_2%20.15f"%coeff[5])


# In[59]:

def plotMag(omega, H, title, xlim, ylim):
    plt.figure(figsize = (11,9))
    H_db = H_db = 20*np.log10(np.abs(H))
    plt.semilogx(omega, H_db, color = "black", linewidth = 2)
    plt.xlim(xlim)
    plt.xlabel("Frequency (Hz)", fontsize = 12)
    plt.ylim(ylim)
    plt.ylabel("Relative level (dB)", fontsize = 12)
    plt.grid(True, which = "both", color = "black", linestyle = (0, (5, 10)))
    plt.title(title + '\n', fontsize = 16)
    #plt.savefig(title + ".pdf")
    plt.show()
    

# In[59]:


def K_filter(signal, fs, debug=False):
    # apply K filtering as specified in EBU R-128 / ITU BS.1770-4

    # pre-filter 1
    f0 = 1681.9744509555319
    G  = 3.99984385397
    Q  = 0.7071752369554193
    K  = np.tan(np.pi * f0 / fs)
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.499666774155)
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K -  Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_
    signal_1 = ss.lfilter([b0,b1,b2],[a0,a1,a2],signal)
    
    if debug:
        print("sample freq:", fs)
        print_coeff([b0,b1,b2,a0,a1,a2],"pre-filter 1")
        omega, H1 = ss.freqz([b0,b1,b2], [a0,a1,a2], worN=10000, fs=fs)
        plotMag(omega, H1, "Response of stage 1", xlim = [10, 20000], ylim = [-10, 10])


    # pre-filter 2
    f0 = 38.13547087613982
    Q  = 0.5003270373253953
    K  = np.tan(np.pi * f0 / fs)
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0
    signal_2 = ss.lfilter([b0,b1,b2],[a0,a1,a2],signal_1)
    
    if debug:
        print_coeff([b0,b1,b2,a0,a1,a2],"pre-filter 2")
        omega, H2 = ss.freqz([b0,b1,b2], [a0,a1,a2], worN=10000, fs=fs)
        plotMag(omega, H2, "Response of stage 2", xlim = [10, 20000], ylim = [-30, 5])

    return signal_2 # return signal passed through 2 pre-filters


# In[49]:


def plot_audio(data):
    data_left = []; data_right = []
    for i in range(len(data)):
        data_left.append(data[i][0])
        data_right.append(data[i][1])

    plt.figure(figsize = (40, 10))
    (markers, stemlines, baseline) = plt.stem(data_left, label='channel_left', linefmt='b-', use_line_collection=True)
    plt.setp(markers, marker="<", markersize=20, markerfacecolor="b")
    (markers, stemlines, baseline) = plt.stem(data_right, label='channel_right', linefmt='r-', use_line_collection=True)
    plt.setp(markers, marker=">", markersize=20, markerfacecolor="r")
    plt.legend(fontsize=25)
    plt.show()


# In[5]:


from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.signal as ss
import pyloudnorm as pyln
import warnings


# In[10]:

fs, data = wavfile.read('input.wav')

print(data.dtype)
if data.dtype == "uint8":
    d = 2**8-1
elif data.dtype == "int32":
    d = 2**31-1
elif data.dtype == "float32":
    d = 1
else:
    d = 2**15-1
print(d)

data = data.astype(np.float)/d

# In[10]:

def peak(data, target):
    """ Peak normalize a signal.
    
    Normalize an input signal to a user specifed peak amplitude.   
    Params
    -------
    data : ndarray
        Input multichannel audio data.
    target : float
        Desired peak amplitude in dB.
    Returns
    -------
    output : ndarray
        Peak normalized output data.
    """
    # find the amplitude of the largest peak
    temp = np.abs(data)
    current_peak = np.max(temp)

    # calculate the gain needed to scale to the desired peak level
    gain = np.power(10.0, target/20.0) / current_peak
    output = gain * data
    
    # check for potentially clipped samples
    if np.max(np.abs(output)) >= 1.0:
        warnings.warn("Possible clipped samples in output.")

    return output
    
# In[10]:
    
def loudness_(data, input_loudness, target_loudness):
    """ Loudness normalize a signal.
    
    Normalize an input signal to a user loudness in dB LKFS.   
    Params
    -------
    data : ndarray
        Input multichannel audio data.
    input_loudness : float
        Loudness of the input in dB LUFS. 
    target_loudness : float
        Target loudness of the output in dB LUFS.
        
    Returns
    -------
    output : ndarray
        Loudness normalized output data.
    """    
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = target_loudness - input_loudness
    gain = np.power(10.0, delta_loudness/20.0)

    output = gain * data

    # check for potentially clipped samples
    if np.max(np.abs(output)) >= 1.0:
        warnings.warn("Possible clipped samples in output.")

    return output

# In[69]:
    
# measure the loudness first 
meter = pyln.Meter(rate=fs, filter_class="DeMan") # create BS.1770 meter
loudness = meter.integrated_loudness(data)
loudness


# In[69]:

# loudness normalize audio to -12 dB LUFS
peak_normalized_audio = peak(data, -12.0)


# In[69]:

# loudness normalize audio to -12 dB LUFS
loudness_normalized_audio = loudness_(data, loudness, -12.0)


# In[69]:

ld1 = calculate_loudness(data, fs)
ld1


# In[61]:


meter = pyln.Meter(fs, filter_class="DeMan")
ld_2 = meter.integrated_loudness(data)
ld_2


# In[9]:

# =============================================================================
# test_data = data[22000:35000]
# pd.DataFrame(test_data)
# =============================================================================


# In[27]:

K_filter(data, fs, True);

