# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_bids

#%matplotlib inline
#%matplotlib qt

# <codecell>

# this works only with mne and mne_bids installed

# <codecell>

path_to_data = "../../BIDS_EEG_meditation_experiment/"

# <codecell>

raw_edf = mne_bids.read_raw_bids("sub-001_ses-01_task-meditation_eeg.bdf", path_to_data)

# <codecell>

raw_edf.plot()

# <codecell>

events = mne.find_events(raw_edf, initial_event=True, consecutive=True)

# <codecell>

events = mne.events_from_annotations(raw_edf)

# <codecell>

raw_edf.plot_psd(fmax=50)
# this works

# <codecell>

raw_edf.plot(events=events)

# <codecell>
