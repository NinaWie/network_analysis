# Network analysis methods for human mobility

The experiments in our paper are based on two datasets. Only the Foursquare data is publicly available. The code published here can be used to reproduce our analysis on the Foursquare dataset. In the folder `data`, we provide the preprocessed data of a small excerpt of the original dataset published in [@yang2015nationtelescope] and [@yang2015participatory] (see below). 

## Installation:

Clone the repository, **with the submodule included**:
```
git clone --recurse-submodules https://github.com/NinaWie/network_analysis.git
```

We assume that the following prerequesites are installed: 
* python >= 3.6
* R (version 4.1.2 is used here)

### R package installation

In R, install renv to create a virtual environment to run the scripts in this repo:
```
install.packages("renv")
```
In the folder `r_scripts` you can find a lockfile (explanation see [here](https://rstudio.github.io/renv/)) called `renv.lock`. The required libraries can be installed automatically with renv by running
```
renv::restore()
```

##### Manual installation
If not using renv, you can also install the required packages manually. The following libraries are required: `sna`, `network`, `RSiena` and can be installed with the following:
```
install.packages("sna")
install.packages("network")
install.packages("RSiena")
```

### Python installation

In Python, create a new virtual environment called "agile_env" and install all requirements with the following steps:
```
cd python_scripts
python -m venv agile_env
source agile_env/bin/activate
pip install -r requirements.txt
export PYTHONPATH="graph-trackintel"
```

Troubleshooting: If the installation of psychopg2 fails, install it manually:
```
python -m pip install psycopg2-binary
````

## Reproducing the results

The following steps must be executed sequentially in order to load, preprocess and analyze the data

### Step 1 : create graphs

Create the graphs for a certain dataset and save as a pickle file in the folder `data`.
```
cd python_scripts
python create_graphs.py
```
Our analysis was run with the current default values for the arguments, but other parameters can be specified. It takes only around 20 seconds to run.

### Step 2: Preprocessing
```
cd python_scripts
python preprocessing.py
```
With the default parameters, this will take the output pkl file from Step 1 and preprocess the graphs and attributes. The results will be dumped in a folder `data/foursquare_120` (120 days).

### Step 3: Execute R scripts

The R Scripts in the folder `r_scripts` automatically take the graphs and attributes that were produced in Step 2. 
Simply execute both scripts `qap_mobility.r` and `saom_mobility.r` (in any order). See above for instructions how to install the necessary libraries with `renv::restore()`

Note: Both scripts will take more than 1 hour to run! SOAM and QAP models are fitted for all users.

### Step 4: Analze results

Run 
```
cd python_scripts
python analyze.py
```
The input folders are hardcoded in the script.

## Acknowledgements:

The Foursquares dataset user for reproducability here was taken from the following sources:

```bib
@article{yang2015nationtelescope,
  title={NationTelescope: Monitoring and visualizing large-scale collective behavior in LBSNs},
  author={Yang, Dingqi and Zhang, Daqing and Chen, Longbiao and Qu, Bingqing},
  journal={Journal of Network and Computer Applications},
  volume={55},
  pages={170--180},
  year={2015},
  publisher={Elsevier}
}

@article{yang2015participatory,
  title={Participatory cultural mapping based on collective behavior in location based social networks},
  author={Yang, Dingqi and Zhang, Daqing and Qu, Bingqing},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2015},
  note = {in press},
  publisher={ACM}
}
```