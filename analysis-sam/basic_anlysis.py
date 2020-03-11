# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mne
#%matplotlib inline

# <codecell>



# <codecell>

raw = mne.io.read_raw_fif('./sam-AlphaTheta/offline/fif/20200304-144100-raw.fif')

# <codecell>

raw.info

# <codecell>

events = mne.find_events(raw, initial_event=True, consecutive=True)

# <codecell>

raw.plot_psd(fmax=50);

# <codecell>

raw.plot_sensors()

# <codecell>

raw.plot(events=events);

# <codecell>

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)

# <codecell>

orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

# show some frontal channels to clearly illustrate the artifact remove

orig_raw.plot(start=12, duration=4)
raw.plot(start=12, duration=4)

# <codecell>

baseline_sam = pd.read_pickle('./sam-AlphaTheta/offline/20200304-144100-raw.pcl')
baseline2_sam = pd.read_pickle('./sam-AlphaTheta/offline/20200304-144601-raw.pcl')

# <codecell>

signals = baseline_sam['signals']

# <codecell>

def plot_signals(data, channels):
    fig, axs = plt.subplots(nrows=data.shape[1], figsize=(40, 1.4 * data.shape[1]))
    for channel in range(data.shape[1]):
        sns.lineplot(data=data[:, channel], ax=axs[channel])
        axs[channel].set_ylabel(channels[channel])
        
    return fig

# <codecell>

baseline_fig = plot_signals(baseline_sam['signals'], baseline_sam['ch_names'])

# <codecell>

fig, axs = plt.subplots(nrows=signals.shape[1], figsize=(40, 1.4 * signals.shape[1]))
for channel in range(signals.shape[1]):
    sns.lineplot(data=signals[:, channel], ax=axs[channel])
    axs[channel].set_ylabel(baseline_sam['ch_names'][channel])

# <codecell>


