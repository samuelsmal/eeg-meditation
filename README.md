# NeuroFeedback for Meditation with EEG-data
## Semester project spring semester 2020, EPFL, MIPLAB

This is the online and crude analysis package.

**NOTE**: Due to legal issues I will not be sharing the sounds. I know that this is a huge problem.
Either drop me an Email or use duckduckgo to search for music and/or sounds.

## Overview of the whole system

See the bellow diagram for a setup of the system. The arrows indicate how the flow of the data.


```
+----------------------------------------+
|EEG Device                              |
|* You will have to start this physically|
+----+-----------------------------------+
     |
     |
     v
+----------------------------------------------------+
|Device Plugin/Driver                                |
|* Establishes a connection to the device            |
|* You will have to start this in a separate terminal|
+----+-----------------------------------------------+
     |
     |
     v
+--------------------------------+
|NeuroDecode                     |
|* Reads the data from the device|
|* Creates the data-streams      |
|* Saves the data to a fif+file  |
+----+---------------------------+
     |
     |
     v
+------------------------------------+
|nfme                                |
|* Reads the data from NeuroDecode   |
|* Uses the data to make magic happen|
|* Controls NeuroDecodoe             |
|* This package                      |
+------------------------------------+
```

## Outline of this package

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

If you're not running a Linux system then you might need to tweak this.

This only works if you have the `conda` package manager installed, get it [here](https://docs.conda.io/en/latest/miniconda.html).

You will have to setup three packages:

1. This one
2. NeuroDecode
3. The "plugin" to interface with the EEG recording device. You will have to get this from Arnaud.

In order to work "out of the box" install them side-by-side in the same directory. Like this:


```
├── device-plugin                                  #
├── NeuroDecode                                    # package 1 (software from arnaud)
└── nfme                                           # package 0 (this one)
```

1. Install basic environment: `conda env create -f environment.yml`
2. Install NeuroDecode (in order to have everything in place for the scripts to run smoothly install
   it side-by-side to the `nfme` package)
  1. Download NeuroDecode: `git clone https://github.com/samuelsmal/NeuroDecode.git`
  2. Add it to the environment: `pip install -e /path/to/NeuroDecode`
3. Activate the environment: `source env`


## Thanks

Supervisor: Raphaël Liégeois, EPFL, MIPLAB
Arnaud Desvachez, Gwénaël Birot from the Campus Biotech for their technical support
