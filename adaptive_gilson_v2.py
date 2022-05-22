"""
2022/02/13
Cassio Amador
Creates adaptive spectrogram, following the papers:
DL JONES, IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 42, NO. 12, DECEMBER 1994

Comments: It seems SLOOOWWWWW...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def create_signal(N):
    # creating sample signal
    N = N if N > 1000 else N
    pulse = np.zeros(N)
    delta = 5
    pos = 100
    pulse[pos:pos + delta] = 1
    pos = 900
    pulse[pos:pos + delta] = 1
    pos = 950
    pulse[pos:pos + delta] = 1

    delta = 200
    pos = 600
    pulse[pos:pos + delta] = signal.gaussian(delta, 25) * np.sin(2 * 60e-3 * np.pi * np.arange(delta))

    delta = 200
    pos = 300
    pulse[pos:pos + delta] = 0.4 * np.sin(2 * 80e-3 * np.pi * np.arange(delta))
    pulse[pos:pos + delta] += 0.4 * np.sin(2 * 40e-3 * np.pi * np.arange(delta))
    #pulse[pos:pos+delta] *= signal.gaussian(delta, 50)
    delta = 700
    if N >= 2000:
        pos = 1200
        pulse[pos:pos + delta] = 0.5 * np.sin(2 * 30e-3 * np.pi * np.arange(delta) * np.linspace(1, 2, num=delta))

    return pulse


def spectrogram_stft(pulse, window=None, nperseg=64, nfft=False):
    window = window or signal.windows.hann(nperseg)
    nfft = nfft or nperseg
    noverlap = nperseg - 1
    f, t, result = signal.stft(pulse, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend='constant')
    result = np.conjugate(result) * result
    return f, t, result.real


def concentration(Dp, window):
    '''Concentration with flat window'''
    C = np.sum(Dp, axis=0)
    c = np.zeros(C.shape)
    for i in range(c.size):
        c[i] = np.sum(C[np.maximum(0, i - window):np.minimum(i + window, c.size)])
    return c


def ghostbusters(time, pulse, ps, mega_window=256):
    ps = np.array(ps)
    f, t, Sadapt = spectrogram_stft(pulse, nperseg=ps[0], nfft=mega_window)
    S = np.zeros((ps.size, Sadapt.shape[0], Sadapt.shape[1]))
    c4s = np.zeros((len(ps), len(pulse) + 1))
    c2s = np.zeros_like(c4s)
    for pp, p in enumerate(ps):
        _, _, S[pp, :, :] = spectrogram_stft(pulse, nperseg=p, nfft=mega_window)

        c4s[pp] = concentration(S[pp, :, :]**4, mega_window // 2)
        c2s[pp] = concentration(S[pp, :, :]**2, mega_window // 2)
        c2s[pp] *= c2s[pp]
        c2s[c2s == 0] = 1

    C = c4s / c2s
    best_win_pos = C.argmax(0)
    best_win = ps[best_win_pos]
    # Sadapt  = [S[best_win_pos[i],:, i] for i in range(best_win_pos.size)]
    for i in range(Sadapt.shape[1]):
        Sadapt[:, i] = S[best_win_pos[i], :, i]
    return C, c4s, c2s, best_win, f, t, Sadapt


N = 2000
time = np.linspace(0, N - 1, N)
pulse = create_signal(N)
ps = np.array([16, 32, 64, 128, 160, 256])
mega_window = 256
C, c4s, c2s, best_win, f, t, Sadapt = ghostbusters(time, pulse, ps, mega_window)
# np.save("best_win", best_win)

ploti = 1
if ploti == 1:
    # pulsinho=signal.detrend(pulse,type="constant")
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    ax[1, 2].contourf(t, f, np.sqrt(Sadapt))
    ax[1, 2].set_ylim(0, 0.125)
    ax[1, 2].set_title(u"adaptive window")
    ax[1, 2].sharey(ax[0, 1])

    ax[0, 0].plot(pulse)
    ax[0, 0].set_title("signal")
    ax[0, 0].set_ylim(-1, 2)
    ax[0, 0].set_xlim(t[0], t[-1])

    win = 16
    f, t, S = spectrogram_stft(pulse, nperseg=win)
    ax[0, 1].contourf(t, f, np.sqrt(S))
    ax[0, 1].set_ylim(0, 0.125)
    ax[0, 1].set_title(f"window: {win}")

    win = 64
    f, t, S = spectrogram_stft(pulse, nperseg=win)
    ax[1, 0].contourf(t, f, np.sqrt(S))
    ax[1, 0].set_ylim(0, 0.125)
    ax[1, 0].set_title(f"window: {win}")
    ax[1, 0].sharey(ax[0, 1])

    win = 256
    f, t, S = spectrogram_stft(pulse, nperseg=win)
    ax[1, 1].contourf(t, f, np.sqrt(S))
    ax[1, 1].set_ylim(0, 0.125)
    ax[1, 1].set_title(f"custom spectrogram: {win}")
    ax[1, 1].sharey(ax[0, 1])

    ax[0, 2].plot(time, best_win[:-1])
    ax[0, 2].set_title('"best" window size')
    # ax[0,2].set_ylim(-1,1)

    plt.tight_layout()
    plt.show()

ploti = 1
if ploti == 2:
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax[0, 0].plot(c2s.T)
    # ax[0, 0].set_ylim(0, 0.125)
    ax[0, 0].set_title(f"C2")
    ax[0, 0].legend(ps, loc="best")

    ax[1, 0].plot(c4s.T)
    # ax[1, 0].set_ylim(0, 0.125)
    ax[1, 0].set_title(f"C4")
    ax[1, 0].legend(ps, loc="best")

    ax[1, 1].plot(C.T)
    # ax[1,1].set_ylim(0, 0.125)
    ax[1, 1].set_title(f"c4/c2")
    ax[1, 1].legend(ps, loc="best")

    ax[0, 1].plot(C[:, 100], label="100")
    ax[0, 1].plot(C[:, 350], label="350")
    ax[0, 1].plot(C[:, 700], label="700")
    # ax[0, 1].set_ylim(0, 0.125)
    ax[0, 1].set_title(f"Cs")
    ax[0, 1].legend(loc="best")
    # ax[1, 0].sharey(ax[0, 1])

    plt.tight_layout()
    plt.show()
