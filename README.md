# NeuroFeedback for Meditation with EEG-data

Semester project spring semester 2020, EPFL, MIPLAB

This is the analysis and system-definition package. Not (at least currently) the online-ready
version.

## Outline

```
.
├── data                                          # example of the data structure
│   ├── AlphaTheta
│   │   └── subject-AlphaTheta                    # for every subject a folder
│   │       ├── online
│   │       └── offline
│   │           ├── 20200304-144100-eve.txt
│   │           ├── 20200304-144100-raw.pcl
│   │           └── fif
│   │               ├── 20200304-144100-raw.fif
│   │               └── channelsList.txt
│   └── bandpowers
├── environment.yml                               # use this to set up the environment
├── nfme
│   ├── config.py                                 # config file
│   ├── features                                  # everything that has to do with features
│   │   ├── bandpower.py
│   │   └── preprocessing.py                      # filtering etc.
│   └── utils                                     # utils in general
│       ├── data_loading.py
│       ├── plots.py
│       └── video.py
├── notebooks                                     # development and main runfiles
│                                                 # you'll find a *.py version of each notebook as
│                                                 # well this is just a copy, making it easier to
│                                                 # track changes and import into other notebooks
└── README.md                                     # this file
```

## Setup

Execute

```
conda env create -f environment.yml
```

## Thanks

Supervisor: Raphaël Liégeois, EPFL, MIPLAB
Arnaud Desvachez, Gwénaël Birot from the Campus Biotech for their technical support
