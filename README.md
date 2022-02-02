# Network analysis methods for human mobility

The experiments in our paper are based on two datasets. Only the Foursquare data is publicly available. The code published here can be used to reproduce our analysis on the Foursquare dataset. In the folder `data`, we provide the preprocessed data of a small excerpt of the original dataset published in [@yang2015nationtelescope] and [@yang2015participatory] (see below). 

### Step 1 : create graphs

Create the graphs for a certain dataset and save as a pickle file as save_name
```
create_graphs.py [-h] [-n NODE_THRESH] [-d DATASET] [-s SAVE_NAME] [-t TIME_PERIOD]
```

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