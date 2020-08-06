
## Online anomaly detection leveraging stream-based clustering and real-time telemetry (ODS)

Scripts used to produce the results in: "Online anomaly detection leveraging stream-based clustering and real-time telemetry"

### Installation
The libraries used in these scripts require python>=3.6

1) You can skip to (Python Virtual Environment) if python>=3.6 and generate a virtual environment.

2) If python<3.6 you have to generate a conda environment and install python 3.6

#### Conda Environment
You can skip these steps if python>=3.6 and generate a python virtual environment 

1) Generate a conda virtual environment `'conda create --name ods python=3.6'`
2) Activate the environment `'conda activate ods'`

#### Python Virtual Environment
Generate a python virtual environment if python>=3.6

Skip these steps if you already generated a conda environment.

1) Generate a python virtual env `'virtualenv ods_env -p python3'`
2) Activate the environment `'source ./ods_env/bin/activate'`

### Libraries Installation

1) Install the libraries `'pip install numpy==1.16.2 pandas==0.24.2 scikit-learn==0.20.2 matplotlib==3.3.0 jupyter==1.0.0 seaborn==0.10.1 fibheap==0.2.1 rrcf=0.4.3'`
2) Move to ods/ and install ods `'python setup.py install'`
5) Move back to the main directory of the repository and run jupyter `'jupyter notebook'`

### Results
To reproduce the results, please follow the instructions. Notice that the repository is organized as follows: 

1) The models and data processing functions are present in Architecture/
2) The data is stored in Data/ while the ground truth can be found in GroundTruth/
3) ods/ contains the source code of ODS. The folder is the clone of ...
4) Experiments/ contains, for each method, the scripts to produce tuning and testing results together with figures.

Figures and Tables:
- Fig. 3 Dataset at a glance: Move to Data/ and follow instructions in readme.
- Fig. 4. Hyperparameter selection. Move to Experiments/ and follow instructions in readme. 
- Fig. 5. ODS Hyperparameter selection. Move to Experiments/ and follow instructions in readme.
- Fig. 6. ODS Hyperparameter selection: Move to Experiments/ and follow instructions in readme. 
- Fig. 7. wDBScan (left) vs ODS (right) model evolution over time: Move to Experiments/ and follow instructions in readme.
- Fig. 8. Algorithms Performance Comparison: Move to Experiments/ and follow instructions in readme.
- Fig. 9. Algorithms Performance Comparison: Move to Experiments/ and follow instructions in readme.
- Fig. 10 - Running Time: Move to Experiments/ and follow instructions in readme.




  
  
  
  
  
  
  
  
  
  
