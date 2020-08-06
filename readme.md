
## Online anomaly detection leveraging stream-based clustering and real-time telemetry (ODS)

Scripts used to produce the results in: "Online anomaly detection leveraging stream-based clustering and real-time telemetry"

### Miniconda Installation
Please skip these steps (go to Environment) if you have CONDA or Miniconda alredy installed. 
1) Install Miniconda Python3.8 following the regular installation instructions: https://conda.io/projects/conda/en/latest/user-guide/install/index.html. You can download the installer from here: https://docs.conda.io/en/latest/miniconda.html
2) Activate miniconda environment. It should automatically activate rebooting the terminal. You can manually activate it with: `'source ~/miniconda3/bin/activate'`
3) Check python version: `'python --version'`. The output should be `'Python 3.8.3'`

### Environment
1) Generate a conda virtual environment `'conda create --name ods_env python=3.8'`
2) Activate the environment `'conda activate ods_env'`

### Packages Installation

1) Install conda packages `'conda install numpy pandas scikit-learn matplotlib jupyter seaborn'` 
2) Install pip packages `'pip install fibheap rrcf'`
2) Move to the main directory of the repository
3) Move to ods/ and install ods `'python setup.py install'`
4) Move back to the main directory of the repository and run jupyter `'jupyter notebook'`

=> Package version: numpy==1.19.1 pandas==1.0.5 scikit-learn==0.23.1 matplotlib==3.2.2 seaborn==0.10.1 jupyter==1.0.0 fibheap==0.2.1 rrcf==0.4.3 ods=0.0.1

### Results
To reproduce the results, please follow the instructions. Notice that the repository is organized as follows: 

1) The models and data processing functions are present in Architecture/
2) The data is stored in Data/ while the ground truth can be found in GroundTruth/
3) ods/ contains the source code of ODS. The folder is the clone of https://github.com/anrputina/ods
4) Experiments/ contains, for each method, the scripts to produce tuning and testing results.

Figures and Tables:
- Fig. 3 Dataset at a glance: Move to Data/ and follow instructions in readme.
- Fig. 4. Hyperparameter selection. Move to Experiments/ and follow instructions in readme. 
- Fig. 5. ODS Hyperparameter selection. Move to Experiments/ and follow instructions in readme.
- Fig. 6. ODS Hyperparameter selection: Move to Experiments/ and follow instructions in readme. 
- Fig. 7. wDBScan (left) vs ODS (right) model evolution over time: Move to Experiments/ and follow instructions in readme.
- Fig. 8. Algorithms Performance Comparison: Move to Experiments/ and follow instructions in readme.
- Fig. 9. Algorithms Performance Comparison: Move to Experiments/ and follow instructions in readme.
- Fig. 10 - Running Time: Move to Experiments/ and follow instructions in readme.
- Table IV - ODS vs RRCF: Move to Experiments and follow instruction in readme


  
  
  
  
  
  
  
  
  
  
