# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import soundfile as sf

# <codecell>

sig, samplerate = sf.read('../meditation_sounds/dvorak_study/script_beat_harmony.wav')

# <codecell>

sig.shape

# <codecell>

samplerate

# <codecell>

import matplotlib.pyplot as plt
plt.plot(sig)

# <codecell>

sig

# <codecell>

sig_script, _ = sf.read('../meditation_sounds/dvorak_study/script.wav')

# <codecell>

sig_without_script

# <codecell>

import numpy as np

# <codecell>

np.max(sig)

# <codecell>

np.mean(sig >= 0, axis=0)

# <codecell>

np.mean(sig_without_script >= 0, axis=0)

# <codecell>

sig_without_script = sig - sig_script

# <codecell>

sf.write('../meditation_sounds/dvorak_study/script_beat_harmony_without_script.wav', sig_without_script, samplerate)

# <codecell>

def add_to_queue(xs, x):
    xs[:-1] = xs[1:]
    xs[-1] = x
    return xs

# <codecell>

q =np.zeros(20)

# <codecell>

add_to_queue(add_to_queue(q, 1), 4)
