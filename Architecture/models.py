# -*- coding: utf-8 -*-
# @Author: andrian
# @Date:   2019-05-29 16:45:41
# @Last Modified by:   Andrian Putina
# @Last Modified time: 2020-08-05 16:51:45

import sys, json, time, copy, os
import numpy as np
import pandas as pd

from statistics import Statistics
from dataManagement import Scaler

from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

import scipy.stats as st
import multiprocessing

from fibheap import *
from mtree import MTree

from ods import Sample, ODS

verbose = 1
scaler_type = 'StandardScaler'

sample_skip_buffer = 30

def exactSTORM(data, window=20, R=0.5, k=5):

	data = tuple(map(tuple, data.values))

	output = np.zeros(len(data))

	mtree = MTree()
	structure = {}
	indexes = {}

	for now, instance in enumerate(data):

		structure[now] = {'count_after':0,
		                  'nn_before':[],
		                  'data': instance}

		indexes[instance] = now

		neighbors_list = mtree.get_nearest(instance, R, k)

		for neighbor in neighbors_list:
			neighbor_index = indexes[neighbor[0]]
			structure[neighbor_index]['count_after'] += 1
			structure[now]['nn_before'].append(neighbor_index)
		    
		mtree.add(instance)
		    
		if now-window in structure:
			departing_instance = structure[now-window]['data']
			del indexes[departing_instance]
			del structure[now-window]
			mtree.remove(departing_instance)
		    
		for query_index in structure:
			prec_neighbors = structure[query_index]['nn_before']
			succ_neighbors = structure[query_index]['count_after']
			if len(prec_neighbors) + succ_neighbors < k:
				output[query_index] = 1
	return output

def COD(data, window_size=20, R=0.1, k=5):
	mtree = MTree()
	scores = np.zeros(data.shape[0])
	data = tuple(map(tuple,data.values))

	structure = {}
	indexes = {}

	anomalies = set()
	fibheap = makefheap()

	for now, instance in enumerate(data):

		if now >= window_size:
			departing_index = now-window_size
			mtree.remove(structure[departing_index]['data'])

			if fibheap.min is None:
				pass
			else:
				x = getfheapmin(fibheap)
				ev = x[0]

				while (ev==now):
					x = fheappop(fibheap)

					index_x = x[1]
					preceding_objects_x = structure[index_x]['Preceding']
					if departing_index in preceding_objects_x:
						preceding_objects_x.remove(departing_index)
					else:
						pass

					if len(preceding_objects) + structure[index_x]['Succeding'] < k:
						anomalies.add(index_x)
					else:
						if len(preceding_objects_x)>0:
							min_exp = np.inf
							for preceding_x in preceding_objects_x:
								if structure[preceding_x]['exp'] > now:
									if structure[preceding_x]['exp'] < min_exp:
										min_exp = structure[preceding_x]['exp']
										ev = min_exp + window_size + 1
							if structure[index_x]['exp'] > now:
								fheappush(fibheap, (ev, index_x))

					if fibheap.min is None:
						break
					else:
						x = getfheapmin(fibheap)
						ev = x[0]


		"""
		Generate new entry for current instance
		"""
		structure[now] = {
		    'Preceding':[],
		    'Succeding':0,
		    'arr': now,
		    'exp': now+window_size,
		    'data': instance
		}
		indexes[instance] = now

		"""
		Make a range query w.r.t. p. Let A the set of objects returned;
		for each q in A
		    nq+ = nq+ 1
		"""
		A = mtree.get_nearest(instance, R, k)

		preceding_objects = []
		for q in A:
			q_index = indexes[q[0]]
			structure[q_index]['Succeding'] += 1
			preceding_objects.append(q_index)

			if (q_index in anomalies) and (len(structure[q_index]['Preceding'])+structure[q_index]['Succeding'] >= k):
				anomalies.remove(q_index)
				if len(structure[q_index]['Preceding'])!= 0:
					preceding_q_list = structure[q_index]['Preceding']
					min_exp = np.inf

					for preceding_q in preceding_q_list:
						if structure[preceding_q]['exp'] < min_exp:
							min_exp = structure[preceding_q]['exp']

					ev = min_exp + window_size + 1
					fheappush(fibheap, (ev, q_index))

			else:
				"""
				Remove from pq object y = min{qi.exp | qi in Pq}
				"""
				preceding_q_list = structure[q_index]['Preceding']
				if len(preceding_q_list) > 0:
					min_exp = np.inf
					min_index = -1
					for preceding_q in preceding_q_list:
						if structure[preceding_q]['exp'] < min_exp:
							min_exp = structure[preceding_q]['exp']
							min_index = preceding_q
					structure[q_index]['Preceding'].remove(min_index)

		structure[now]['Preceding'] = preceding_objects
		if len(preceding_objects) < k:
			"""
			Add p to D(R,K)
			"""
			anomalies.add(now)
		else:
			"""
			ev = min{pi.exp|pi in Pp}
			insert(p, ev+[W/Slide])
			"""
			min_exp = np.inf
			for preceding_p in preceding_objects:
			    if structure[preceding_p]['exp'] < min_exp:
			        min_exp = structure[preceding_p]['exp']
			ev = min_exp + window_size + 1
			fheappush(fibheap, (ev, now))
		"""
		Add p to data structure supporting range queries
		"""
		mtree.add(instance)

	scores[list(anomalies)] = 1
	return scores

def get_features_node(node, rootPath, features='ALL'):
	features_node = json.loads(open(rootPath+'features_node.json').read())

	if features == 'ALL':
		features_to_use = features_node[node]['DataPlane']+features_node[node]['ControlPlane']
	elif features == 'DP':
		features_to_use = features_node[node]['DataPlane']
	else:
		sys.exit('Unkown features set')
	features_to_use = features_to_use+['time']
	return features_to_use

def read_data(datasets, rootPath, features_set):
	if verbose>0:
		print('Loading Data ... ')

	result = {}

	for dataset in datasets:
		config = json.loads(open(rootPath+'configuration.json').read())['datasets'][dataset]
		for node in config['nodes']:
			features_node = get_features_node(node, rootPath, features_set)
			df = pd.read_csv(rootPath + config['directory']+node+config['filename'],
			     low_memory=False, dtype='float64', compression='gzip')
			df = df[features_node]

			times = df['time']//1e9
			times = times.astype('int')
			df.drop(['time'], axis=1, inplace=True)
			sampleSkip = sample_skip_buffer

			scaler = Scaler(scaler_type)
			dfNormalized = scaler.normalize(df)
			if verbose > 1:
				print('Dataset: {} - Node: {} - Shape: {}'.format(dataset,node,df.shape))            

			bufferDF = dfNormalized
			testDF = dfNormalized

			bufferDF_ods = dfNormalized[0:sampleSkip]
			testDF_ods = dfNormalized[sampleSkip:]			

			node_result = {
			    'times':times,
			    'buffer':bufferDF,
			    'test':testDF,
			    'buffer_ods':bufferDF_ods,
			    'test_ods':testDF_ods,
			    'dataset':dataset,
			    'node':node,
			    'sampleSkip':sampleSkip
			}

			result[dataset+'_'+node] = node_result

	if verbose>0:
		print('Loading Data Done')

	return result

def load_tuning_configuration(filename):
	tuning_list = json.loads(open(filename).read())['tuning_list']
	tuning_configuration = {}

	for dataset in tuning_list:
		dataset_config = json.loads(open(filename).read())['datasets'][dataset]
		tuning_configuration[dataset] = dataset_config

	return tuning_configuration

def read_tuning_configuration(filename='../configuration.json'):
	try:
		return json.loads(open(filename).read())['tuning_configuration']
	except Exception as e:
		raise e

def read_test_configuration(filename='../configuration.json'):
	try:
		return json.loads(open(filename).read())['test_configuration']
	except Exception as e:
		raise e

class Model(object):
	"""docstring for Model"""
	def __init__(self, model='dbscan', config=None, rootPath=None, features_set=None, MAX_PROCESSES=4):
		super(Model, self).__init__()
		self.model = model
		self.features_set = features_set
		self.MAX_PROCESSES = MAX_PROCESSES
		print('Model: {}'.format(self.model))

		if config is None:
			sys.exit('You should pass a config file with directories')
		else:
			self.config = config
			print('Config: {}'.format(self.config))

		if rootPath is None:
			sys.exit('You should pass a rootPath')
		else:
			self.rootPath = rootPath
			print('rootPath: {}'.format(rootPath))

		if self.model == 'dbscan':
			self.parameters = {
				'epsilon': np.arange(1.0, 20.1, 1).astype(float),
				'min_samples': np.arange(2,51,2).astype(int)
			}
		elif self.model == 'wdbscan':
			self.parameters = {
				'epsilon': np.arange(1.0, 20.1, 1).astype(float),
				'min_samples': np.arange(2,51,2).astype(int),
				'window': np.arange(20,100,5).astype(int)
			}
		elif self.model == 'lof':
			self.parameters = {
				'n_neighbors' : np.arange(1,50,1).astype(int),
				'leaf_size' :  [30],
				'contamination': np.arange(0.001, 0.2, 0.005).astype(float)
			}
		elif self.model == 'exactSTORM':
			self.parameters = {
				'window': np.arange(10, 105, 5).astype(int),
				'radius': np.arange(1.0, 20.1, 1).astype(float),
				'k': np.arange(2, 10, 1).astype(int)
			}
		elif self.model == 'COD':
			self.parameters = {
				'k': np.arange(2,10,1).astype(int),
				'window': np.arange(10,105,5).astype(int),
				'radius': np.arange(1.0, 20.1, 1).astype(float)
			}
		elif self.model == 'rrcf':
			self.parameters = {
				'tree_size': np.arange(5, 105, 5).astype(int),
				'contamination': np.arange(0.001, 0.2, 0.005).astype(float)
			}	
		elif self.model == 'ods':
			self.parameters = None
		else:
			sys.exit('Unkown model parameters')

		print('Parameters: {}'.format(self.parameters))

	def load_best_parameters(self):
		self.best_parameters = json.loads(open(self.rootPath+'Experiments/'+self.model+'/Results/'+self.model+'_best_parameters.json').read())

	def set_best_parameters(self, parameters):
		self.best_parameters = parameters

	def get_node_data(self, task, ods=False):

		if not ods:

			st = Statistics(task['dataset'], self.rootPath)
			node_data_buffer = copy.deepcopy(self.data[task['dataset']+'_'+task['node']]['buffer'])
			node_data_test = copy.deepcopy(self.data[task['dataset']+'_'+task['node']]['test'])
			times = copy.deepcopy(self.data[task['dataset']+'_'+task['node']]['times'])

		else:
			st = Statistics(task['dataset'], self.rootPath)
			node_data_buffer = copy.deepcopy(self.data[task['dataset']+'_'+task['node']]['buffer_ods'])
			node_data_test = copy.deepcopy(self.data[task['dataset']+'_'+task['node']]['test_ods'])
			times = copy.deepcopy(self.data[task['dataset']+'_'+task['node']]['times'])			

		return st, node_data_buffer, node_data_test, times

	def worker(self, task):
		if verbose>0:
			print('Task: {}'.format(task))

		st, node_data_buffer, node_data_test, times = self.get_node_data(task)

		if self.model == 'dbscan':
			if verbose>1:
				print(node_data_test.shape)

			epsilon = task['epsilon']
			min_samples = task['min_samples']

			db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(node_data_test)

			model_result = pd.Series(list(db.labels_))
			model_result = model_result.apply(lambda x: False if x >=0 else True)

		elif self.model == 'wdbscan':

			epsilon = task['epsilon']
			min_samples = task['min_samples']
			window = task['window']

			labels = []

			samples = node_data_test[:window]
			db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(samples)
			labels.extend(list(db.labels_))

			for index in range(node_data_test.shape[0]+1):

				if index > window:
					samples = node_data_test.iloc[index-window:index]
					db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(samples)
					labels.append(db.labels_[-1])

			model_result = pd.Series(list(labels))
			model_result = model_result.apply(lambda x: False if x >=0 else True)

		elif self.model == 'lof':

			n_neighbors = task['n_neighbors']
			leaf_size = task['leaf_size']
			contamination = task['contamination']

			lof = LocalOutlierFactor(n_neighbors=int(n_neighbors),
									contamination=contamination,
									leaf_size=leaf_size)

			model_result = pd.Series(lof.fit_predict(node_data_test))
			model_result = model_result.apply(lambda x: False if x >=0 else True)

		elif self.model == 'exactSTORM':
			window = task['window']
			radius = task['radius']
			k = task['k']

			model_result = exactSTORM(
				node_data_test,
				window=window,
				R=radius,
				k=k
			)
			model_result = pd.Series(model_result).apply(lambda x: False if x==0 else True)
			model_result[:sample_skip_buffer] = False

		elif self.model == 'COD':

			window = task['window']
			R = task['R']
			k = task['k']

			model_result = COD(
				node_data_test,
				window_size=window,
				R=R,
				k=k
			)

			model_result = pd.Series(model_result).apply(lambda x: False if x==0 else True)
			model_result[:sample_skip_buffer] = False

		elif self.model == 'rrcf':
			import rrcf

			points = node_data_test.values
			tree_size = task['tree_size']
			contamination = task['contamination']

			num_trees = 100

			# Create a forest of empty trees
			forest = []
			for _ in range(num_trees):
				tree = rrcf.RCTree()
				forest.append(tree)
			# Create a dict to store anomaly score of each point
			avg_codisp = {}

			for index, point in enumerate(points):
				# For each tree in the forest...
				for tree in forest:
					# If tree is above permitted size, drop the oldest point (FIFO)
					if len(tree.leaves) > tree_size:
						tree.forget_point(index - tree_size)
					# Insert the new point into the tree
					tree.insert_point(point, index=index)
					# Compute codisp on the new point and take the average among all trees
					if not index in avg_codisp:
						avg_codisp[index] = 0
					avg_codisp[index] += tree.codisp(index) / num_trees

			scores = np.array(list(avg_codisp.values()))
			scores_sorted = -np.sort(-scores)
			threshold = scores_sorted[int(np.ceil(contamination*len(scores)))]
			model_result = scores>threshold

			model_result[:sample_skip_buffer] = False
			model_result = pd.Series(model_result)

		elif self.model == 'ods':

			st, node_data_buffer, node_data_test, times = self.get_node_data(task, ods=True)

			epsilon = task['epsilon']
			mu = 'auto'
			tp = 'auto'

			lamb = float(task['lambda'])
			beta = float(task['beta'])
			k_std = float(task['k_std'])

			ods = ODS(
						lamb = lamb,\
			            epsilon = epsilon,\
			            beta = beta,\
			            mu = mu,\
			            startingBuffer = node_data_buffer.values,
			            tp = tp,
			            k_std=k_std)
			ods.runInitialization()


			startingSimulation = time.time()
			outputCurrentNode = []
			for sampleNumber in range(len(node_data_test)):
			    sample = node_data_test.iloc[sampleNumber]
			    result = ods.runOnNewSample(Sample(sample.values, times.iloc[sampleNumber]))[0]
			    outputCurrentNode.append(result)
			endSimulation = time.time() - startingSimulation				

			model_result = pd.Series([False] * node_data_buffer.shape[0] + outputCurrentNode)

		else:
			sys.exit('Unkown model worker')

		KT = 1
		st.getScores(times, model_result, KT)

		task['precision'] = st.gt.Precision
		task['recall'] = st.gt.Recall
		task['fscore'] = st.gt.Fscore
		task['TNR'] = st.gt.TNR
		task['NPV'] = st.gt.NPV
		task['FNR'] = st.gt.FNR
		task['FDR'] = st.gt.FDR
		task['FOR'] = st.gt.FOR
		task['TS'] = st.gt.TS
		task['PT'] = st.gt.PT
		task['FPR'] = st.gt.FPR
		task['ACC'] = st.gt.Accuracy
		task['MCC'] = st.gt.MCC
		task['MK'] = st.gt.MK
		task['BM'] = st.gt.BM
		task['FM'] = st.gt.FM

		return task

	def get_tasks(self, task_type='tune'):
		
		tasks = []
		task_id = 0

		if task_type == 'tune':

			if self.model == 'dbscan':

				for epsilon in self.parameters['epsilon']:
					for min_samples in self.parameters['min_samples']:
						for name_current_data, curren_data in self.data.items():
							tasks.append({
								'dataset':curren_data['dataset'],
								'node':curren_data['node'],
								'epsilon': epsilon,
								'min_samples': min_samples,
								'id': task_id
								})
							task_id += 1

			elif self.model == 'wdbscan':

				for epsilon in self.parameters['epsilon']:
					for min_samples in self.parameters['min_samples']:
						for window in self.parameters['window']:
							for name_current_data, curren_data in self.data.items():
								tasks.append({
										'dataset':curren_data['dataset'],
										'node':curren_data['node'],
										'epsilon': epsilon,
										'min_samples': min_samples,
										'window': window,
										'id': task_id
									})
								task_id+=1

			elif self.model == 'lof':

				for n_neighbors in self.parameters['n_neighbors']:
					for leaf_size in self.parameters['leaf_size']:
						for contamination in self.parameters['contamination']:
							for name_current_data, curren_data in self.data.items():
								tasks.append({
										'dataset':curren_data['dataset'],
										'node':curren_data['node'],
										'n_neighbors': n_neighbors,
										'leaf_size':leaf_size,
										'contamination':contamination,
										'id': task_id
									})
								task_id+=1

			elif self.model == 'exactSTORM':

				for window in self.parameters['window']:
					for radius in self.parameters['radius']:
						for k in self.parameters['k']:
							for name_current_data, curren_data in self.data.items():
								tasks.append({
										'dataset':curren_data['dataset'],
										'node':curren_data['node'],
										'window': window,
										'radius': radius,
										'k':k,
										'id': task_id										
									})
								task_id+=1

			elif self.model == 'COD':
				for window in self.parameters['window']:
					for radius in self.parameters['radius']:
						for k in self.parameters['k']:
							for name_current_data, curren_data in self.data.items():
								tasks.append({
										'dataset':curren_data['dataset'],
										'node': curren_data['node'],
										'window': window,
										'R': radius,
										'k': k,
										'id': task_id
									})
								task_id+=1
	
			elif self.model == 'rrcf':
				for tree_size in self.parameters['tree_size']:
					for contamination in self.parameters['contamination']:
						for name_current_data, curren_data in self.data.items():
							tasks.append({
									'dataset':curren_data['dataset'],
									'node': curren_data['node'],
									'tree_size': tree_size,
									'contamination': contamination,
									'id': task_id
								})
							task_id += 1
			else:
				sys.exit('Unkown model for tasks')


			if verbose>0:
				print('Total #tasks: {}'.format(len(tasks)))

			return tasks

		elif task_type == 'test':

			if self.model == 'dbscan':

				for name_current_data, curren_data in self.data.items():

					if self.features_set == 'ALL':
						tasks.append({
							'dataset':curren_data['dataset'],
							'node':curren_data['node'],
							'epsilon': 6,
							'min_samples': 18,
							'id':task_id						
							})
						task_id+=1
					else:
						sys.exit('Unkown features_set')					

			elif self.model == 'wdbscan':
				
				for name_current_data, curren_data in self.data.items():
					if self.features_set == 'ALL':
						tasks.append({
							'dataset':curren_data['dataset'],
							'node':curren_data['node'],
							'epsilon': 9,
							'min_samples': 3,
							'window': 80,
							'id':task_id						
							})
						task_id+=1
					else:
						sys.exit('Unkown features_set')

			elif self.model == 'lof':
				
				for name_current_data, curren_data in self.data.items():

					if self.features_set == 'ALL':
						tasks.append({
								'dataset':curren_data['dataset'],
								'node':curren_data['node'],
								'n_neighbors': 24,
								'leaf_size': 30,
								'contamination':0.065,
								'id':task_id							
							})
						task_id+=1
					else:
						sys.exit('Unkown features_set')

			elif self.model == 'exactSTORM':
				for name_current_data, curren_data in self.data.items():

					if self.features_set == 'ALL':
						tasks.append({
								'dataset':curren_data['dataset'],
								'node': curren_data['node'],
								'window': 95,
								'radius': 10,
								'k': 2
							})
	
			elif self.model == 'COD':
				for name_current_data, curren_data in self.data.items():
					if self.features_set == 'ALL':
						tasks.append({
								'dataset': curren_data['dataset'],
								'node': curren_data['node'],
								'k': 5,
								'R': 12.5,
								'window': 50
							})
			elif self.model == 'rrcf':
					for name_current_data, curren_data in self.data.items():
						if self.features_set == 'ALL':
							tasks.append({
									'dataset': curren_data['dataset'],
									'node': curren_data['node'],
									'tree_size': 95,
									'contamination':0.03
								})

			elif self.model == 'ods':

				for name_current_data, curren_data in self.data.items():
					tasks.append({
							'dataset':curren_data['dataset'],
							'node':curren_data['node'],
							'lambda': float(self.best_parameters['lambda']),
							'beta': float(self.best_parameters['beta']),
							'k_std': float(self.best_parameters['k_std']),
							'epsilon': self.best_parameters['epsilon'],
							'id':task_id					
						})
			else:
				sys.exit('Unkown model tasks')


			return tasks

		else:
			sys.exit('Unkown task_type: {}'.format(task_type))

	def get_best(self, df, kpi = 'fscore'):

		if self.model == 'dbscan':

			scores = df.groupby(['epsilon','min_samples'])[kpi].mean()
			best_epsilon, best_min_samples = scores.idxmax()

			res = {}
			res['epsilon'] = float(best_epsilon)
			res['min_samples'] = int(best_min_samples)
			res['score'] = float(scores.max())

			return pd.Series(res, index=['epsilon', 'min_samples', 'score'])

		elif self.model == 'wdbscan':

			scores = df.groupby(['epsilon', 'min_samples', 'window'])[kpi].mean()
			best_epsilon, best_min_samples, best_window = scores.idxmax()

			res = {}
			res['epsilon'] = float(best_epsilon)
			res['min_samples'] = int(best_min_samples)
			res['window'] = int(best_window)
			res['score'] = float(scores.max())

			return pd.Series(res, index=['epsilon', 'min_samples', 'window', 'score'])

		elif self.model == 'lof':

			scores = df.groupby(['n_neighbors', 'leaf_size', 'contamination'])[kpi].mean()
			best_n_neighbors, best_leaf_size, best_contamination = scores.idxmax()

			res = {
				'n_neighbors': int(best_n_neighbors),
				'leaf_size': int(best_leaf_size),
				'contamination': best_contamination,
				'score': float(scores.max())
			}

			return pd.Series(res, index=['n_neighbors', 'leaf_size', 'contamination', 'score'])

		elif self.model == 'exactSTORM':
			scores = df.groupby(['window', 'radius', 'k'])[kpi].mean()
			best_window, best_radius, best_k = scores.idxmax()
			res = {
				'window':int(best_window),
				'radius': float(best_radius),
				'k': int(best_k),
				'score': float(scores.max())
			}
			return pd.Series(res, index=['window', 'radius', 'k'])

		else:
			sys.exit('Unkown best score')	

	def get_mean_conf_interval(self,df):
		res = {}

		precision_key = 'precision'
		recall_key = 'recall'
		fscore_key = 'fscore'
		TNR_key = 'TNR'
		NPV_key = 'NPV'
		FNR_key = 'FNR'
		FDR_key = 'FDR'
		FOR_key = 'FOR'
		TS_key = 'TS'
		PT_key = 'PT'
		FPR_key = 'FPR'
		ACC_key = 'ACC'
		MCC_key = 'MCC'
		MK_key = 'MK'
		BM_key = 'BM'
		FM_key = 'FM'

		res['avg_prec'] = df[precision_key].mean()
		res['avg_rec'] = df[recall_key].mean()
		res['avg_fscore'] = df[fscore_key].mean()
		res['avg_TNR'] = df[TNR_key].mean()
		res['avg_NPV'] = df[NPV_key].mean()
		res['avg_FNR'] = df[FNR_key].mean()
		res['avg_FDR'] = df[FDR_key].mean()
		res['avg_FOR'] = df[FOR_key].mean()
		res['avg_TS'] = df[TS_key].mean()
		res['avg_PT'] = df[PT_key].mean()
		res['avg_FPR'] = df[FPR_key].mean()
		res['avg_ACC'] = df[ACC_key].mean()
		res['avg_MCC'] = df[MCC_key].mean()
		res['avg_MK'] = df[MK_key].mean()
		res['avg_BM'] = df[BM_key].mean()
		res['avg_FM'] = df[FM_key].mean()
		    
		res['ci_prec_low'] = res['avg_prec'] - st.t.interval(0.95, len(df[precision_key].dropna())-1, loc=df[precision_key].dropna().mean(), scale=st.sem(df[precision_key].dropna()))[0]
		res['ci_prec_high'] = st.t.interval(0.95, len(df[precision_key].dropna())-1, loc=df[precision_key].dropna().mean(), scale=st.sem(df[precision_key].dropna()))[1] - res['avg_prec']

		res['ci_rec_low'] = res['avg_rec'] - st.t.interval(0.95, len(df[recall_key])-1, loc=df[recall_key].mean(), scale=st.sem(df[recall_key]))[0] 
		res['ci_rec_high'] = st.t.interval(0.95, len(df[recall_key])-1, loc=df[recall_key].mean(), scale=st.sem(df[recall_key]))[1] - res['avg_rec'] 

		res['ci_fscore_low'] = res['avg_fscore'] - st.t.interval(0.95, len(df[fscore_key])-1, loc=df[fscore_key].mean(), scale=st.sem(df[fscore_key]))[0] 
		res['ci_fscore_high'] = st.t.interval(0.95, len(df[fscore_key])-1, loc=df[fscore_key].mean(), scale=st.sem(df[fscore_key]))[1] - res['avg_fscore'] 
	    
		res['ci_TNR_low'] = res['avg_TNR'] - st.t.interval(0.95, len(df[TNR_key])-1, loc=df[TNR_key].mean(), scale=st.sem(df[TNR_key]))[0] 
		res['ci_TNR_high'] = st.t.interval(0.95, len(df[TNR_key])-1, loc=df[TNR_key].mean(), scale=st.sem(df[TNR_key]))[1] - res['avg_TNR'] 

		res['ci_NPV_low'] = res['avg_NPV'] - st.t.interval(0.95, len(df[NPV_key])-1, loc=df[NPV_key].mean(), scale=st.sem(df[NPV_key]))[0] 
		res['ci_NPV_high'] = st.t.interval(0.95, len(df[NPV_key])-1, loc=df[NPV_key].mean(), scale=st.sem(df[NPV_key]))[1] - res['avg_NPV'] 

		res['ci_FNR_low'] = res['avg_FNR'] - st.t.interval(0.95, len(df[FNR_key])-1, loc=df[FNR_key].mean(), scale=st.sem(df[FNR_key]))[0] 
		res['ci_FNR_high'] = st.t.interval(0.95, len(df[FNR_key])-1, loc=df[FNR_key].mean(), scale=st.sem(df[FNR_key]))[1] - res['avg_FNR'] 

		res['ci_FDR_low'] = res['avg_FDR'] - st.t.interval(0.95, len(df[FDR_key])-1, loc=df[FDR_key].mean(), scale=st.sem(df[FDR_key]))[0] 
		res['ci_FDR_high'] = st.t.interval(0.95, len(df[FDR_key])-1, loc=df[FDR_key].mean(), scale=st.sem(df[FDR_key]))[1] - res['avg_FDR']

		res['ci_FOR_low'] = res['avg_FOR'] - st.t.interval(0.95, len(df[FOR_key])-1, loc=df[FOR_key].mean(), scale=st.sem(df[FOR_key]))[0] 
		res['ci_FOR_high'] = st.t.interval(0.95, len(df[FOR_key])-1, loc=df[FOR_key].mean(), scale=st.sem(df[FOR_key]))[1] - res['avg_FOR']

		res['ci_TS_low'] = res['avg_TS'] - st.t.interval(0.95, len(df[TS_key])-1, loc=df[TS_key].mean(), scale=st.sem(df[TS_key]))[0] 
		res['ci_TS_high'] = st.t.interval(0.95, len(df[TS_key])-1, loc=df[TS_key].mean(), scale=st.sem(df[TS_key]))[1] - res['avg_TS']

		res['ci_PT_low'] = res['avg_PT'] - st.t.interval(0.95, len(df[PT_key])-1, loc=df[PT_key].mean(), scale=st.sem(df[PT_key]))[0] 
		res['ci_PT_high'] = st.t.interval(0.95, len(df[PT_key])-1, loc=df[PT_key].mean(), scale=st.sem(df[PT_key]))[1] - res['avg_PT']

		res['ci_FPR_low'] = res['avg_FPR'] - st.t.interval(0.95, len(df[FPR_key])-1, loc=df[FPR_key].mean(), scale=st.sem(df[FPR_key]))[0] 
		res['ci_FPR_high'] = st.t.interval(0.95, len(df[FPR_key])-1, loc=df[FPR_key].mean(), scale=st.sem(df[FPR_key]))[1] - res['avg_FPR']

		res['ci_ACC_low'] = res['avg_ACC'] - st.t.interval(0.95, len(df[ACC_key])-1, loc=df[ACC_key].mean(), scale=st.sem(df[ACC_key]))[0] 
		res['ci_ACC_high'] = st.t.interval(0.95, len(df[ACC_key])-1, loc=df[ACC_key].mean(), scale=st.sem(df[ACC_key]))[1] - res['avg_ACC']

		res['ci_MCC_low'] = res['avg_MCC'] - st.t.interval(0.95, len(df[MCC_key])-1, loc=df[MCC_key].mean(), scale=st.sem(df[MCC_key]))[0] 
		res['ci_MCC_high'] = st.t.interval(0.95, len(df[MCC_key])-1, loc=df[MCC_key].mean(), scale=st.sem(df[MCC_key]))[1] - res['avg_MCC']		

		res['ci_MK_low'] = res['avg_MK'] - st.t.interval(0.95, len(df[MK_key])-1, loc=df[MK_key].mean(), scale=st.sem(df[MK_key]))[0] 
		res['ci_MK_high'] = st.t.interval(0.95, len(df[MK_key])-1, loc=df[MK_key].mean(), scale=st.sem(df[MK_key]))[1] - res['avg_MK']

		res['ci_BM_low'] = res['avg_BM'] - st.t.interval(0.95, len(df[BM_key])-1, loc=df[BM_key].mean(), scale=st.sem(df[BM_key]))[0] 
		res['ci_BM_high'] = st.t.interval(0.95, len(df[BM_key])-1, loc=df[BM_key].mean(), scale=st.sem(df[BM_key]))[1] - res['avg_BM']

		res['ci_FM_low'] = res['avg_FM'] - st.t.interval(0.95, len(df[FM_key])-1, loc=df[FM_key].mean(), scale=st.sem(df[FM_key]))[0] 
		res['ci_FM_high'] = st.t.interval(0.95, len(df[FM_key])-1, loc=df[FM_key].mean(), scale=st.sem(df[FM_key]))[1] - res['avg_FM']

		return pd.Series(res, index=['avg_prec', 'avg_rec', 'avg_fscore', 'avg_TNR', 'avg_NPV', 'avg_FNR',
									'avg_FDR', 'avg_FOR', 'avg_TS', 'avg_PT', 'avg_FPR', 'avg_ACC', 'avg_MCC',
									'avg_MK', 'avg_BM', 'avg_FM',
									'ci_prec_low', 'ci_prec_high',
									'ci_rec_low', 'ci_rec_high', 
									'ci_fscore_low', 'ci_fscore_high',
									'ci_TNR_low', 'ci_TNR_high',
									'ci_NPV_low', 'ci_NPV_high',
									'ci_FNR_low', 'ci_FNR_high',
									'ci_FDR_low', 'ci_FDR_high',
									'ci_FOR_low', 'ci_FOR_high',
									'ci_TS_low', 'ci_TS_high',
									'ci_PT_low', 'ci_PT_high',
									'ci_FPR_low', 'ci_FPR_high',
									'ci_ACC_low', 'ci_ACC_high',
									'ci_MCC_low', 'ci_MCC_high',
									'ci_MK_low', 'ci_MK_high',
									'ci_BM_low', 'ci_BM_high',
									'ci_FM_low', 'ci_FM_high'
									])

	def get_general_scores(self, df):

		res = {}

		precision_key = 'precision'
		recall_key = 'recall'
		fscore_key = 'fscore'
		TNR_key = 'TNR'
		NPV_key = 'NPV'
		FNR_key = 'FNR'
		FDR_key = 'FDR'
		FOR_key = 'FOR'
		TS_key = 'TS'
		PT_key = 'PT'
		FPR_key = 'FPR'
		ACC_key = 'ACC'
		MCC_key = 'MCC'
		MK_key = 'MK'
		BM_key = 'BM'
		FM_key = 'FM'

		res['precision'] = df[precision_key].dropna().mean()
		res['recall'] = df[recall_key].mean()
		res['fscore'] = df[fscore_key].dropna().mean()

		res['TNR'] = df[TNR_key].mean()
		res['NPV'] = df[NPV_key].mean()
		res['FNR'] = df[FNR_key].mean()
		res['FDR'] = df[FDR_key].mean()
		res['FOR'] = df[FOR_key].mean()
		res['TS'] = df[TS_key].mean()
		res['PT'] = df[PT_key].mean()
		res['FPR'] = df[FPR_key].mean()
		res['ACC'] = df[ACC_key].mean()
		res['MCC'] = df[MCC_key].mean()
		res['MK'] = df[MK_key].mean()
		res['BM'] = df[BM_key].mean()
		res['FM'] = df[FM_key].mean()

		res['ci_precision'] = res['precision'] - st.t.interval(0.95, len(df[precision_key].dropna())-1, loc=df[precision_key].dropna().mean(), scale=st.sem(df[precision_key].dropna()))[0]
		res['ci_recall'] = res['recall'] - st.t.interval(0.95, len(df[recall_key])-1, loc=df[recall_key].mean(), scale=st.sem(df[recall_key]))[0] 
		res['ci_fscore'] = res['fscore'] - st.t.interval(0.95, len(df[fscore_key].dropna())-1, loc=df[fscore_key].dropna().mean(), scale=st.sem(df[fscore_key].dropna()))[0] 
		res['ci_TNR'] = res['TNR'] - st.t.interval(0.95, len(df[TNR_key])-1, loc=df[TNR_key].mean(), scale=st.sem(df[TNR_key]))[0]
		res['ci_NPV'] = res['NPV'] - st.t.interval(0.95, len(df[NPV_key])-1, loc=df[NPV_key].mean(), scale=st.sem(df[NPV_key]))[0]
		res['ci_FNR'] = res['FNR'] - st.t.interval(0.95, len(df[FNR_key])-1, loc=df[FNR_key].mean(), scale=st.sem(df[FNR_key]))[0]
		res['ci_FDR'] = res['FDR'] - st.t.interval(0.95, len(df[FDR_key])-1, loc=df[FDR_key].mean(), scale=st.sem(df[FDR_key]))[0]
		res['ci_FOR'] = res['FOR'] - st.t.interval(0.95, len(df[FOR_key])-1, loc=df[FOR_key].mean(), scale=st.sem(df[FOR_key]))[0]
		res['ci_TS'] = res['TS'] - st.t.interval(0.95, len(df[TS_key])-1, loc=df[TS_key].mean(), scale=st.sem(df[TS_key]))[0]
		res['ci_PT'] = res['PT'] - st.t.interval(0.95, len(df[PT_key])-1, loc=df[PT_key].mean(), scale=st.sem(df[PT_key]))[0]
		res['ci_FPR'] = res['FPR'] - st.t.interval(0.95, len(df[FPR_key])-1, loc=df[FPR_key].mean(), scale=st.sem(df[FPR_key]))[0]
		res['ci_ACC'] = res['ACC'] - st.t.interval(0.95, len(df[ACC_key])-1, loc=df[ACC_key].mean(), scale=st.sem(df[ACC_key]))[0]
		res['ci_MCC'] = res['MCC'] - st.t.interval(0.95, len(df[MCC_key])-1, loc=df[MCC_key].mean(), scale=st.sem(df[MCC_key]))[0]
		res['ci_MK'] = res['MK'] - st.t.interval(0.95, len(df[MK_key].dropna())-1, loc=df[MK_key].dropna().mean(), scale=st.sem(df[MK_key].dropna()))[0]
		res['ci_BM'] = res['BM'] - st.t.interval(0.95, len(df[BM_key])-1, loc=df[BM_key].mean(), scale=st.sem(df[BM_key]))[0]
		res['ci_FM'] = res['FM'] - st.t.interval(0.95, len(df[FM_key])-1, loc=df[FM_key].mean(), scale=st.sem(df[FM_key]))[0]

		self.general_scores = res

	def run_grid_search(self):

		self.data = read_data(self.config['datasets'], self.rootPath, self.features_set)	
		tasks = self.get_tasks(task_type='tune')

		p = multiprocessing.Pool(processes=self.MAX_PROCESSES)
		
		results = p.map_async(self.worker, tasks)

		p.close()
		p.join()

		self.totalRes = []

		for r in results.get():
			self.totalRes.append(r)

		self.totalRes = pd.DataFrame(self.totalRes)

		beta = 0.5
		self.totalRes['fscore'] = (1+beta*beta)*self.totalRes['precision']*self.totalRes['recall']/(beta*beta*self.totalRes['precision']+self.totalRes['recall'])
		
		directory = 'Results/'
		if not os.path.exists(directory):
			os.makedirs(directory)

		self.totalRes.to_csv(directory+self.model+'_totalRes_'+self.features_set+'.csv', index=None)


	def run(self):
		self.data = read_data(self.config['datasets'], self.rootPath, self.features_set)
		tasks = self.get_tasks(task_type='test')

		p = multiprocessing.Pool(processes=self.MAX_PROCESSES)

		results = p.map_async(self.worker, tasks)

		p.close()
		p.join()

		self.totalRes = []

		for r in results.get():
			self.totalRes.append(r)

		self.totalRes = pd.DataFrame(self.totalRes)

		self.results = self.totalRes.groupby('dataset').apply(self.get_mean_conf_interval)
		self.results = self.results.fillna(0)

		directory = 'Results/'
		if not os.path.exists(directory):
			os.makedirs(directory)

		self.totalRes.to_csv(directory+self.model+'_all_nodes_results_test_'+self.features_set+'.csv', index=None)
		self.results.to_json(directory+self.model+'_results_test_'+self.features_set+'.json', orient='index')

		self.get_general_scores(self.totalRes[self.totalRes['dataset'].isin(['BGP_testbed_5', 'BGP_testbed_9', 'BGP_testbed_10'])])

		with open(directory+self.model+'_general_results_test_'+self.features_set+'.json', 'w') as outfile:
			json.dump(self.general_scores, outfile, indent=2)

		self.get_general_scores(self.totalRes[self.totalRes['dataset'].isin(['BGP_testbed_2', 'BGP_testbed_3'])])

		with open(directory+self.model+'_general_results_tune_'+self.features_set+'.json', 'w') as outfile:
			json.dump(self.general_scores, outfile, indent=2)