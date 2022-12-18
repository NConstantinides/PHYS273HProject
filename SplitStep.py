import numpy as np
from matplotlib import pyplot as plt, cm

from collections.abc import Iterable
from abc import ABC, abstractmethod

plot = plt.plot()


def CutoffRange(a):
    return np.argmax(a), a.shape[-1] - np.argmax(a[::-1]) - 1


def CutoffFilter(a):
    nMin, nMax = CutoffRange(a)

    filt = np.arange(0, len(a), 1)
    filt = (filt >= nMin) & (filt <= nMax)
    return filt


class PulseBase(ABC):
    @abstractmethod
    def evaluate(self, time):
        pass


class NLSE:
    class GuassianPulse(PulseBase):
        def __init__(self, amplitude, spread, offset):
            self.amplitude = amplitude
            self.spread = spread
            self.offset = offset

        def evaluate(self, time):
            return self.amplitude * np.exp(-((time - self.offset) / self.spread)**2 / 2)

    class SechPulse(PulseBase):
        def __init__(self, t0, amplitude):
            self.t0 = t0
            self.amplitude = amplitude

        def evaluate(self, time):
            return self.amplitude/np.cosh(time / self.t0)

    class SimParams:
        def __init__(self, N, dt, dz):
            self.N = N
            self.dt = dt
            self.dz = dz
            self.t = np.linspace(-N*dt/2, N*dt/2, N)
            self.freq = np.fft.fftshift(np.fft.fftfreq(N, dt))

    class FiberParams:
        def __init__(self, beta2, alpha, gamma, length):
            self.beta2 = beta2
            self.beta2ZDep = callable(beta2)
            self.alpha = alpha
            self.gamma = gamma
            self.length = length

    def TimeToSpectralDomain(self):
        return np.fft.fftshift(np.fft.fft(self.A)) * self.sim.dt

    def SpectralToTimeDomain(self):
        return np.fft.ifft(np.fft.ifftshift(self.AFreq)) / self.sim.dt

    def __init__(self, simParams: SimParams, fiberParams: FiberParams, pulse: PulseBase):
        self.sim = simParams
        self.fiber = fiberParams
        self.A = np.cdouble(pulse.evaluate(self.sim.t))
        self.AFreq = self.TimeToSpectralDomain()

    """
    Operates in the time domain and applies the non-linear effects of self phase modulation
    """

    def OperatorN(self):
        self.A *= np.exp(1j * self.fiber.gamma *
                         np.abs(self.A)**2 * self.sim.dz)

    """
    Operates in the frequency domain and applies the non-linear effects of group velocity dispersion / attenuation
    """

    def OperatorD(self, z):
        self.AFreq = self.TimeToSpectralDomain()
        beta2 = self.fiber.beta2 if not self.fiber.beta2ZDep else self.fiber.beta2(z)
        self.AFreq *= np.exp((1j * beta2 / 2 * (2 * np.pi * self.sim.freq)**2
                             - self.fiber.alpha / 2) * self.sim.dz)
        self.A = self.SpectralToTimeDomain()

    def Evolve(self):
        zsteps = np.int64(np.ceil(self.fiber.length / self.sim.dz))
        self.ATimes = np.ndarray((zsteps, self.A.shape[-1]),
                                 dtype=self.A.dtype)
        self.AFreqs = np.ndarray((zsteps, self.AFreq.shape[-1]),
                                 dtype=self.AFreq.dtype)

        for i in range(zsteps):
            self.ATimes[i] = np.copy(self.A)
            self.AFreqs[i] = np.copy(self.AFreq)
            self.OperatorN()
            self.OperatorD(i * self.sim.dz)

    def PlotPowerEvolution3D(self, fig, ax3D, powerCutoff=0.001):
        power = np.abs(self.ATimes[-1])**2
        filt = CutoffFilter(power > np.max(power) * powerCutoff)
        aSliced = self.ATimes[:, filt]
        tSliced = self.sim.t[filt] * 1e12 # put in picoseconds
        z = np.arange(0, self.fiber.length, self.sim.dz)

        tMesh, zMesh = np.meshgrid(tSliced, z)
        ax3D.set_xlabel("T (ps)")
        ax3D.set_ylabel("Z (m)")
        ax3D.set_zlabel("P (W)")
        ax3D.set_title("Power vs Time")
        ax3D.plot_surface(tMesh, zMesh, np.abs(aSliced)**2, cmap=cm.viridis,
                          linewidth=0, antialiased=False)

    def PlotPower1D(self, fig, ax2D, z: Iterable, powerCutoff=0.001):
        n = np.array(np.floor(np.array(z) / self.sim.dz), dtype=np.int64)
        power = np.abs(self.ATimes[n.max()])**2
        filt = CutoffFilter(power > np.max(power) * powerCutoff)
        ax2D.set_xlabel("T (ps)")
        ax2D.set_ylabel("P (w)")
        ax2D.set_title("Power vs Time")
        for i in n:
            ax2D.plot(self.sim.t[filt] * 1e12, # put in picoseconds
                      np.abs(self.ATimes[i, filt])**2, # watts
                      label = f"z = {i * self.sim.dz}"
            )
        fig.legend()

    def PlotFrequencyEvolution1D(self, fig, ax2D, z: Iterable, cutoffRatio = 0.01):
        n = np.array(np.floor(np.array(z) / self.sim.dz), dtype=np.int64)
        maxAmp = np.max(np.abs(self.AFreqs[n.max()])**2)
        cutoffAmp = maxAmp * cutoffRatio
        filt = CutoffFilter(np.abs(self.AFreqs[n.max()])**2 > cutoffAmp)
        freq = self.sim.freq[filt] / 1e9 # put in gigahertz units
        amps = np.abs(self.AFreqs[n][:,filt])**2 * 1e9 # put in W/gigahertz
        ax2D.set_xlabel("f (GHz)")
        ax2D.set_ylabel("PSD (W/GHz)")
        ax2D.set_title("Power Spectral Density")
        for i in range(n.shape[-1]):
            ax2D.plot(freq, amps[i], label = f'z = {n[i] * self.sim.dz}')
        fig.legend()

    def PlotFrequencyEvolution2D(self, fig, ax2D, cutoffRatio = 0.01):
        maxAmp = np.max(np.abs(self.AFreqs[-1])**2)
        cutoffAmp = maxAmp * cutoffRatio
        filt = CutoffFilter(np.abs(self.AFreqs[-1])**2 > cutoffAmp)
        freq = self.sim.freq[filt] / 1e9 # put in gigahertz units
        amps = np.abs(self.AFreqs[:,filt])**2 * 1e9 # put in W/gigahertz
        F, Z = np.meshgrid(freq, np.arange(0, self.fiber.length, self.sim.dz))
        surf = ax2D.contourf(F, Z, amps)
        ax2D.set_xlabel("f (GHz)")
        ax2D.set_ylabel("z (m)")
        ax2D.set_title("Power Spectral Density")
        fig.colorbar(surf, ax = ax2D, label="PSD (W/GHz)")
    
    def PlotFrequencyEvolutionDecibels2D(self, fig, ax2D, cutoffRatio = 0.01, dBMin=-32):
        maxAmp = np.max(np.abs(self.AFreqs[-1])**2)
        cutoffAmp = maxAmp * cutoffRatio
        filt = CutoffFilter(np.abs(self.AFreqs[-1])**2 > cutoffAmp)
        freq = self.sim.freq[filt] / 1e9 # put in gigahertz units
        amps = np.abs(self.AFreqs[:,filt])**2
        amps /= np.max(amps)
        amps[amps<1e-100]=1e-100 # set a low cuttoff to make the graph coloring nice
        amps = 10 * np.log10(amps)
        amps[amps < dBMin] = dBMin
        F, Z = np.meshgrid(freq, np.arange(0, self.fiber.length, self.sim.dz))
        surf = ax2D.contourf(F, Z, amps)
        ax2D.set_xlabel("f (GHz)")
        ax2D.set_ylabel("z (m)")
        ax2D.set_title("Spectral Intensity")
        fig.colorbar(surf, ax = ax2D, label="I (dB/GHz)")

    def PlotChirpVals(self, fig, ax2D, z: Iterable, tAxis = None):
        tAxis = tAxis if tAxis != None else (-3e-11*1e12, 3e-11*1e12)
        phase = np.unwrap(np.angle(self.ATimes))
        dPhase = np.diff(phase)
        dT = np.diff(self.sim.t)
        chirp = -1.0/(2*np.pi)*dPhase/dT
        ax2D.set_xlabel("T (ps)")
        ax2D.set_ylabel("chirp (GHz)")
        ax2D.set_title("Pulse Chirp")
        for i in z:
            ax2D.plot(self.sim.t[1:] * 1e12, # put in picosecs
                chirp[np.int64(np.floor(i / self.sim.dz))] / 1e9, # gigahertz,
                label = f'z = {i}'
            )
        ax2D.set_xlim(tAxis)
        ax2D.set_ylim((-50, 50))
        fig.legend()

    def PlotAll(self, file_prefix = "plots/plot", file_postfix = ".png", zlocs = None, show = False):
        zlocs = zlocs if zlocs != None else [0, self.fiber.length - self.sim.dz]

        figPowerFL, axPowerFL = plt.subplots(1, 1, figsize=(5, 4))
        figPower3D, axPower3D = plt.subplots(1, 1, figsize = (5, 5), 
            subplot_kw = dict(projection = "3d"))
        figFreqPFL, axFreqPFL = plt.subplots(1, 1, figsize = (5, 4))
        figFreqPFL2D, axFreqPFL2D = plt.subplots(1, 1, figsize = (5, 4))
        figFreqDbFL2D, axFreqDbFL2D = plt.subplots(1, 1, figsize = (5, 4))
        figChirpFL, axChirpFL = plt.subplots(1, 1, figsize = (5, 4))

        self.PlotPowerEvolution3D(figPower3D, axPower3D)
        self.PlotPower1D(figPowerFL, axPowerFL, zlocs)
        self.PlotFrequencyEvolution1D(figFreqPFL, axFreqPFL, zlocs)
        self.PlotFrequencyEvolution2D(figFreqPFL2D, axFreqPFL2D)
        self.PlotFrequencyEvolutionDecibels2D(figFreqDbFL2D, axFreqDbFL2D)
        self.PlotChirpVals(figChirpFL, axChirpFL, zlocs)
        
        figPowerFL.savefig(f'{file_prefix}PowerFL{file_postfix}')
        figPower3D.savefig(f'{file_prefix}Power3D{file_postfix}')
        figFreqPFL.savefig(f'{file_prefix}FreqPFL{file_postfix}')
        figFreqPFL2D.savefig(f'{file_prefix}FreqPFL2D{file_postfix}')
        figFreqDbFL2D.savefig(f'{file_prefix}FreqDbFL2D{file_postfix}')
        figChirpFL.savefig(f'{file_prefix}ChirpFL{file_postfix}')

        if show:
            plt.show()

        
    
def main():
    params = NLSE.SimParams(
        2**14,                           # Number of discrete time data points
        0.1e-12,                         # Time resolution in seconds
        1000/2**8                        # Fiber position resolution in meters
    )
    fiber = NLSE.FiberParams(
        35 * 1e-30 * 1e3,                # Dispersion parameter in s^2/m
        0.2e-3 * np.log(10)/10.0,        # Attenuation parameter in decibels/m
        1.5e-3,                          # Non-linearity parameter in 1/W/m
        2000                             # Fiber length in meters
    )
    pulse = NLSE.GuassianPulse(
        1,                               # Square root of the peak power in watts
        2**7*params.dt,                  # Spread in units of 1/e^2 * seconds
        0                                # Offset in second
    )

    sechParams = NLSE.SimParams(
        2**16,                           # Number of discrete time data points
        0.1e-12,                         # Time resolution in seconds
        2000/2**8                        # Fiber position resolution in meters
    )
    sechFiber = NLSE.FiberParams(
        20 * 1e-24 * 1e-3,               # Dispersion parameter in s^2/m
        0.2e-3 * np.log(10)/10.0,        # Attenuation parameter in decibels/m
        3e-3,                            # Non-linearity parameter in 1/W/m
        1000                             # Fiber length in meters
    )
    sechPulse = NLSE.SechPulse(
        2**6 * sechParams.dt,
        np.sqrt(0.64 * 1e-3) # power in mW
    )

    combBeta = lambda z: 20 * 1e-24 * 1e-3 if 0 <= z < 1000 else -80 * 1e-24 * 1e-3
    compParams = NLSE.SimParams(
        2**14,                           # Number of discrete time data points
        0.1e-12,                         # Time resolution in seconds
        1250/2**8                        # Fiber position resolution in meters
    )
    compFiber = NLSE.FiberParams(
        combBeta,                        # Dispersion parameter in s^2/m
        0.2e-3 * np.log(10)/10.0,        # Attenuation parameter in decibels/m
        1.5e-3,                          # Non-linearity parameter in 1/W/m
        1250                             # Fiber length in meters
    )
    compPulse = NLSE.GuassianPulse(
        1,                               # Square root of the peak power in watts
        2**7*params.dt,                  # Spread in units of 1/e^2 * seconds
        0                                # Offset in second
    )

    simulation = NLSE(params, fiber, pulse)
    simulation.Evolve()
    simulation.PlotAll()

    sechSim = NLSE(sechParams, sechFiber, sechPulse)
    sechSim.Evolve()
    sechSim.PlotAll("plots/sech")

    compSim = NLSE(compParams, compFiber, compPulse)
    compSim.Evolve()
    compSim.PlotAll("plots/comp", 
        zlocs = (0, 1000, 1250 - compParams.dz))




if __name__ == "__main__":
    main()
