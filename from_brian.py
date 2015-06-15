from brian import *
from numpy import *
import numpy
import array as pyarray
import time
import struct
try:
    import pygame
    have_pygame = True
except ImportError:
    have_pygame = False
try:
    from scikits.samplerate import resample
    have_scikits_samplerate = True
except (ImportError, ValueError):
    have_scikits_samplerate = False
from bufferable import Bufferable
from prefs import get_samplerate
from db import dB, dB_type, dB_error, gain
from scipy.signal import fftconvolve, lfilter
from scipy.misc import factorial

    def save(self, filename, normalise=False, samplewidth=2):
        '''
        Save the sound as a WAV.
        
        If the normalise keyword is set to True, the amplitude of the sound will be
        normalised to 1. The samplewidth keyword can be 1 or 2 to save the data as
        8 or 16 bit samples.
        '''
        ext = filename.split('.')[-1].lower()
        if ext=='wav':
            import wave as sndmodule
        elif ext=='aiff' or ext=='aifc':
            import aifc as sndmodule
            raise NotImplementedError('Can only save as wav soundfiles')
        else:
            raise NotImplementedError('Can only save as wav soundfiles')
        
        if samplewidth != 1 and samplewidth != 2:
            raise ValueError('Sample width must be 1 or 2 bytes.')
        
        scale = {2:2 ** 15, 1:2 ** 7-1}[samplewidth]
        if ext=='wav':
            meanval = {2:0, 1:2**7}[samplewidth]
            dtype = {2:int16, 1:uint8}[samplewidth]
            typecode = {2:'h', 1:'B'}[samplewidth]
        else:
            meanval = {2:0, 1:2**7}[samplewidth]
            dtype = {2:int16, 1:uint8}[samplewidth]
            typecode = {2:'h', 1:'B'}[samplewidth]
        w = sndmodule.open(filename, 'wb')
        w.setnchannels(self.nchannels)
        w.setsampwidth(samplewidth)
        w.setframerate(int(self.samplerate))
        x = array(self,copy=True)
        am=amax(x)
        z = zeros(x.shape[0]*self.nchannels, dtype=x.dtype)
        x.shape=(x.shape[0],self.nchannels)
        for i in range(self.nchannels):
            if normalise:
                x[:,i] /= am
            x[:,i] = (x[:,i]) * scale + meanval
            z[i::self.nchannels] = x[::1,i]
        data = array(z, dtype=dtype)
        data = pyarray.array(typecode, data)
        w.writeframes(data.tostring())
        w.close()

    def spectrogram(self, low=None, high=None, log_power=True, other = None,  **kwds):
        '''
        Plots a spectrogram of the sound
        
        Arguments:
        
        ``low=None``, ``high=None``
            If these are left unspecified, it shows the full spectrogram,
            otherwise it shows only between ``low`` and ``high`` in Hz.
        ``log_power=True``
            If True the colour represents the log of the power.
        ``**kwds``
            Are passed to Pylab's ``specgram`` command.
        
        Returns the values returned by pylab's ``specgram``, namely
        ``(pxx, freqs, bins, im)`` where ``pxx`` is a 2D array of powers,
        ``freqs`` is the corresponding frequencies, ``bins`` are the time bins,
        and ``im`` is the image axis.
        '''
        if self.nchannels>1:
            raise ValueError('Can only plot spectrograms for mono sounds.')
        if other is not None:
            x = self.flatten()-other.flatten()
        else:
            x = self.flatten()
        pxx, freqs, bins, im = specgram(x, Fs=self.samplerate, **kwds)
        if low is not None or high is not None:
            restricted = True
            if low is None:
                low = 0*Hz
            if high is None:
                high = amax(freqs)*Hz
            I = logical_and(low <= freqs, freqs <= high)
            I2 = where(I)[0]
            I2 = [max(min(I2) - 1, 0), min(max(I2) + 1, len(freqs) - 1)]
            Z = pxx[I2[0]:I2[-1], :]
        else:
            restricted = False
            Z = pxx
        if log_power:
            Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
            Z = 10 * log10(Z)
        Z = flipud(Z)
        if restricted:
            imshow(Z, extent=(0, amax(bins), freqs[I2[0]], freqs[I2[-1]]),
                   origin='upper', aspect='auto')
        else:
            imshow(Z, extent=(0, amax(bins), freqs[0], freqs[-1]),
                   origin='upper', aspect='auto')
        xlabel('Time (s)')
        ylabel('Frequency (Hz)')
        return (pxx, freqs, bins, im)