Fig. 4. Hyperparameter selection. The plots are produced i) running the hyperparameter grid search first and ii) generating the figure afterwards:

- DBScan
	1. Move to dbscan/
	2. Run [dbscan_grid_search.ipynb] to test the parameters. The results are saved in ./dbscan/Results/. 
	3. NOTE: 1) The process is very long. 2) It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will) 
- LOF
	1. Move to lof/
	2. Run [lof_grid_search.ipynb] to test the parameters. The results are saved in: ./lof/Results/
	3. NOTE: 1) The process is very long. 2) It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will)    
- wDBScan
	1. Move to wdbscan/
	2. Run [wdbscan_grid_search.ipynb] to test the parameters. The results are saved in ./wdbscan/Results/
	3. NOTE: 1) The process is very long. 2) It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will) 
- ExactSTORM
	1. Move to exacstorm/
	2. Run [wdbscan_grid_search.ipynb] to test the parameters. The results are saved in ./exactstorm/Results/
	3. NOTE: 1) The process is very long. 2) It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will)     
- COD
	1. Move to COD/
	2. Run [wdbscan_grid_search.ipynb] to test the parameters. The results are saved in ./COD/Results/
	3. NOTE: 1) The process is very long. 2) It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will) 
- RRCF
	1. Move to RRCF/
	2. Run [wdbscan_grid_search.ipynb] to test the parameters. The results are saved in ./RRCF/Results/
	3. NOTE: 1) The process is very long. 2) It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will)   
  
- Generate Figure
	1. Move to main directory.
	2. Move to Results/GridHeatmap/
	3. Run [dbscan_wdbdscan_lof_grid.ipynb]. The script loads the results from Experiments/dbscan/Results/, Experiments/wdbscan/Results/ and Experiments/lof/Results and generates the figure.

Fig. 5. ODS Hyperparameter selection:
  1. Move to ods/ParametersSelection/
  2. Run [k_std.ipynb] to test the impact of kr and generate the figure. The figure is saved in ./ods/Results/
  3. NOTE: It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will)
  
Fig. 6. ODS Hyperparameter selection:
  1. Move to ods/ParametersSelection/
  2. Run [lambda_vs_beta.ipynb] to test the impact of lambda and beta. The figure is saved in ./ods/Results/
  3. NOTE: It uses multiprocessing - MAX_PROCESSES tasks in parallel. MAX_PROCESSES=30 (modify at will)

Fig. 7. wDBScan (left) vs ODS (right) model evolution over time: The plots are produced i) running the algorithms and exporting the variables first and ii) generating the figure afterwards:
  1. Move to z-model-evolution/
  2. Run [wdbscan_model_evolution.ipynb] which runs the model and exports the variables of wdbscan in ./z-model-evolution/wDBScan_variables/
  3. Run [ods_model_evolution.ipynb] which runs the model and exports the variables of ODS in ./z-model-evolution/ODS_variables/
  4. Run [plot_models_evolution.ipynb] which reads the variables and saves the figure in ./z-model-evolution/Figures/
  
 Fig. 8 and Fig. 9 Algorithms Performance Comparison: The plots are produced i) running the algorithms first and ii) generating the figure afterwards.
 
  - DBScan
    1. Move to dbscan/
    2. Run dbscan_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./dbscan/Results/
  - LOF
    1. Move to lof/
    2. Run lof_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./lof/Results/    
  - wDBScan
    1. Move to wdbscan/
    2. Run wdbscan_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./wdbscan/Results/
  - ExactSTORM
    1. Move to exactstorm/
    2. Run exactstorm_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./exactstorm/Results/
  - COD
    1. Move to cod/
    2. Run cod_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./cod/Results/
  - RRCF
    1. Move to rrcf/
    2. Run rrcf_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./rrcf/Results/
  - ODS:
    1. Move to Experiments/ods/
    2. Run ods_test.ipynb which runs the model over all the datasets/experiments and generates the results in ./Results/
    
  - Generate Figures:
    1. Move to z2-test_results/
    2. Run [parse_results.ipynb] which parse the results previously generated and outputs total_results_ALL_PRF/AMI.json
    3. Run [generate_figures.ipynb] to generate Fig 8 and Fig 9 from total_results_ALL_PRF/AMI.json

Fig. 10 and Fig. 11 - Running Time and Scatter Plot. The results are produced by i) generating the dataset, ii) running the execution time measurements, and iii) generating the figure.
  1. Generate dataset
    	1. Move to z3-complexity_analysis/ExecutionTime
    	2. Run [generate_dataset.ipynb] to generate the dataset
  2. Run execution time measurements 
    	1. Run [dbscan_complexity.ipynb]
    	2. Run [lof_complexity.ipynb]
    	3. Run [wdbscan_complexity.ipynb]
    	4. Run [exactSTORM_complexity.ipynb]
    	5. Run [cod_complexity.ipynb]
    	6. Run [rrcf_complexity.ipynb]    
    	7. Run [ods_complexity.ipynb]
    	8. => The outputs are saved in ./Results
  3. Generate the figure
    	1. Move back to Experiments/z3-complexity_analysis/
    	2. Run [running_time_plot.ipynb] which loads the measurements from ExecutionTime/ and generates Fig. 10 and Fig. 11
	
Table IV - ODS vs RRCF: 
  1. Move to z4-ods_vs_rrcf_scores/
  2. Run [run_ods_rrcf_scores.ipynb] which runs the models and saves the anonaly scores in ods_scores.pkl and rrcf_scores.pkd
  3. Run [parse_scores.ipynb] to load the scores and process them.
