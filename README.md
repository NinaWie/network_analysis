# Network analysis methods for human mobility

The experiments in our paper are based on two datasets. Only the Foursquare data is publicly available. The code published here can be used to reproduce our analysis on the Foursquare dataset. In the folder `data`, we provide the preprocessed data of a small excerpt of the original dataset published in [@yang2015nationtelescope] and [@yang2015participatory] (see below). 

### Step 1 : create graphs

Create the graphs for a certain dataset and save as a pickle file as save_name
```
cd python_scripts
create_graphs.py [-h] [-n NODE_THRESH] [-d DATASET] [-s SAVE_NAME] [-t TIME_PERIOD]
```
Our analysis was run with the current default values for the arguments.

### Step 2: Preprocessing
```
cd python_scripts
preprocessing.py [-h] [-o OUT_DIR] [-i INPUT] [-t TIME_BINS]
```
With the default parameters, this will take the output pkl file from Step 1 and preprocess the graphs and attributes. The results will be dumped in a folder `data/foursquare_120` (120 days).

### Step 3: Execute R scripts

The R Scripts in the folder `r_scripts` automatically take the graphs and attributes that were produced in Step 2. 
Execute both scripts `r_scripts/qap_mobility.r` and `r_scripts/saom_mobility.r`

### Step 4: Analze results

Run 
```
cd python_scripts
python analyze.py
```
The input folders are hardcoded in the script.

## The Foursquare dataset:

The material published here was taken from the following sources. 

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