import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import os
import seaborn as sns

__all__ = ['read_freq',
           'PEMFC_equivalent_circuits',
           'zf_cal',
           'zf_check',
           'plot_Nyquist']

def read_freq():
    #path = os.getcwd()
    #freq = np.loadtxt(path+'\PEMFC_freq.txt')
    freq = np.array([5.0000e+04, 3.1548e+04, 1.9905e+04, 1.2559e+04, 7.9245e+03,
       5.0000e+03, 3.1548e+03, 1.9905e+03, 1.2559e+03, 7.9245e+02,
       5.0000e+02, 3.1548e+02, 1.9905e+02, 1.2559e+02, 7.9245e+01,
       5.0000e+01, 3.1548e+01, 1.9905e+01, 1.2559e+01, 7.9245e+00,
       5.0000e+00, 3.1548e+00, 1.9905e+00, 1.2559e+00, 7.9245e-01,
       5.0000e-01, 3.1548e-01, 1.9905e-01, 1.2559e-01, 7.9245e-02,
       5.0000e-02])
    return freq

def circuits_components_params(eis,freq=read_freq()):
    from impedance.models.circuits import CustomCircuit
    customCircuit = CustomCircuit(initial_guess=[1e-9, .001, 5, 0.6, .005,100, 0.005], 
                                  circuit='L_0-R_0-p(CPE_1,R_1)-p(C_2,R_2)')
    
    nums = eis.shape[0]
    params = pd.DataFrame(index=range(nums),columns=['L','Rm','Q','phi','Rct','Cdl','Rt'])
    #customCircuits = []
    len_freq = len(freq)
    for i in range(nums):
        Z = eis.iloc[i,:len_freq].values - (1j)*eis.iloc[i,len_freq:].values
        customCircuit.fit(freq, Z)
        customCircuit_fit = customCircuit.predict(freq)
        #customCircuits.append(customCircuit)
        params.loc[i] = customCircuit.parameters_
    
    return params

def PEMFC_equivalent_circuits(L,Rm,Q,phi,Rct,C,Rt,omega):
    
    zL = (1j)*omega*L
    zcpe=1/(Q*(1j*omega)**phi)
    zc = 1/(1j*omega*C)
    zf = zL + Rm + (Rct*zcpe)/(Rct+zcpe) + (Rt*zc)/(Rt+zc)
    return zf

def zf_cal(L,Rm,Q,phi,Rct,C,Rt,freq=read_freq()):
    # Calculate impendance spectrometry of 60 intervals in the range from 20KHz to 0.02Hz  

    zf_real =[]
    zf_imag =[]
    for i in range(31):
        f = freq[i]
        omega = 2*np.pi*f
        zf = PEMFC_equivalent_circuits(L,Rm,Q,phi,Rct,C,Rt,omega)
        zf_real.append(zf.real)
        zf_imag.append((-1)*zf.imag)
    return zf_real,zf_imag


def zf_check(zf):
    #print(len(zf))
    if len(zf) == 2 :
        zf_real = zf[0]
        zf_imag = zf[1]
    else :
        beg = int(len(zf)%2)
        size = int(len(zf)%2+len(zf)/2)
        zf_real = zf[beg:size]
        zf_imag = zf[size:]
##    if len(zf_real)!=len(zf_imag) :
##        size = np.min(np.size(zf_real),np.size(zf_imag))
##        zf_real = zf_real[:size]
##        zf_imag = zf_imag[:size]
    return zf_real, zf_imag

def plot_Nyquist(z_example,labels,annotate=False,freq=read_freq()):
# Illustration for 2. Equivalent circuit

        
    nums = len(z_example)

    if nums == 0 :
        print('There is no data!')
    if nums == 1:
        z_real, z_imag = zf_check(z_example[0])
        plt.scatter(z_real,z_imag,label=labels[0],s=5)
        x_pos = z_real
        y_pos = z_imag
        if annotate :
            for i in range(31):
                plt.annotate('{:.1e}'.format(freq[i]),
                             xy=(x_pos[i], y_pos[i]),
                             xytext=(10,-10), textcoords='offset points',
                             ha='left', c='k', va='top',size=2)
    if nums > 1 :   
        for i in range(nums) :
            z_real, z_imag = zf_check(z_example[i])
            plt.scatter(z_real,z_imag,label=labels[i],s=5)
        if annotate and i==0:
            x_pos = z_real
            y_pos = z_imag
            for i in range(31):
                plt.annotate('{:.1e}'.format(freq[i]),
                             xy=(x_pos[i], y_pos[i]),
                             xytext=(10,-10), textcoords='offset points',
                             ha='left', c='k', va='top',size=2)
    plt.legend(bbox_to_anchor=(1.15, 1.),loc='upper right', borderaxespad=0)
    plt.xlabel('[real]')
    plt.ylabel('[-imag]')
    #plt.show()


def plot_Bode(z_example,stois,freq=read_freq()):
    labels = []
    lg_freq = np.log10(freq)
    i=0
    fig,[ax1,ax2] = plt.subplots(2,1,figsize=(6, 6))
    palette = sns.color_palette("bright",8)
    for s in stois:
        labels.append('O2_stoi=%0.1f'%s)
        Z = z_example[:31]-z_example[31:]*1j
        mag = np.abs(Z)
        phase =-np.angle(Z,deg=True)
        ax1.plot(lg_freq,mag,c=palette[i],label=s)
        ax2.plot(lg_freq, phase,c=palette[i],label=s)
        i+=1
    ax1.set_ylabel('|Z|/Ω $\mathregular{m^2}$', fontsize=14)
    ax2.set_ylabel(' $\mathregular{-\phi/\circ}$', fontsize=14)
    ax2.set_yticks(np.arange(-60,60,20))
    string_labels = [r"$10^{%2d}$" % i for i in np.arange(-2,6,1)]
    ax2.set_xticks(np.arange(-2,6,1))
    ax2.set_xlabel('log10(Freq [Hz])', fontsize=14)
    ax1.legend(labels=labels,
               loc='upper right',markerscale=1.2,frameon=False, fontsize=10)
    ax1.set_xticks(np.arange(-2,6,1))
