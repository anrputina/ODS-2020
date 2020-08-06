# -*- coding: utf-8 -*-
# @Author: andrian
# @Date:   2019-04-02 14:29:29
# @Last Modified by:   andrian
# @Last Modified time: 2019-12-30 16:36:13

import sys
import json
import numpy as np
import pandas as pd

debug = False

class GroundTruth(object):
	"""
	Ground Truth class. Useful to compare results with ground truth for all the datasets used

	:param datasetName: the 'name' of the dataset used
	"""
	def __init__(self, datasetName, configurationFilePath, **kwargs):
		super(GroundTruth, self).__init__()
		self.datasetName = datasetName

		self.configurationFilePath = configurationFilePath

		if datasetName == 'BGP_testbed_5':
			self.loadGroundTruthBGP_testbed()

		elif datasetName == 'BGP_testbed_6':
			self.loadGroundTruthBGP_testbed()			

		elif datasetName == 'BGP_testbed_4':
			self.loadGroundTruthBGP_testbed()

		elif datasetName == 'BGP_testbed_2':
			self.loadGroundTruthBGP_testbed()

		elif datasetName == 'BGP_testbed_3':
			self.loadGroundTruthBGP_testbed()

		elif datasetName == 'BGP_testbed_9':
			self.loadGroundTruthBGP_testbed()

		elif datasetName == 'BGP_testbed_10':
			self.loadGroundTruthBGP_testbed()	

		elif datasetName == 'BGP_VIRL_8':
			self.loadGroundTruthBGP_testbed()						

		else:
			sys.exit('Please select a dataset available in configuration: GroundTruth')

		self.tp = 0
		self.tn = 0
		self.fn = 0
		self.fp = 0

		self.Precision = None
		self.Recall = None
		self.Accuracy = None
		self.F1 = None

	def loadGroundTruthBGP_testbed(self):
		self.configuration = json.loads(open(self.configurationFilePath+'configuration.json').read())['datasets'][self.datasetName]
		df = pd.read_csv(self.configurationFilePath + self.configuration['groundtruth'])
		df['End'] = df['Start'] + self.configuration['eventEND']
		df['previousEnd'] = df['End'].shift().fillna(0)
		self.df = df

	def computeScores(self, times, algorithmOutput, KT=3):
		if self.datasetName == 'BGP_testbed_5':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)
		if self.datasetName == 'BGP_testbed_6':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)			
		elif self.datasetName == 'BGP_testbed_4':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)
		elif self.datasetName == 'BGP_testbed_2':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)
		elif self.datasetName == 'BGP_testbed_3':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)	
		elif self.datasetName == 'BGP_testbed_9':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)	
		elif self.datasetName == 'BGP_testbed_10':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)	
		elif self.datasetName == 'BGP_VIRL_8':
			return self.computeScoresBGP_testbed(times, algorithmOutput, KT)										
		else:
			sys.exit('ERROR')

	def computeScoresBGP_testbed(self, times, algorithmOutput, KT=3):
		eventsList = self.df.to_dict('records')
		currentTimestamp = 0
		status = 'normal'

		if debug:
			print('KT={}'.format(KT))

		self.tp = 0
		self.tn = 0
		self.fn = 0
		self.fp = 0

		for event in eventsList:

			normalIndex = times[(times<event['Start']) & (times>event['previousEnd'])].index
			anomalousIndex = times[(times>=event['Start']) & (times<=event['End'])].index

			if debug:
				print('Event: {}'.format(event))
				print('There are {} normal samples'.format(normalIndex.shape))
				print('There are {} anomalous samples.'.format(anomalousIndex.shape))

			"""
			Check normal interval
			"""

			consecutiveAnomalousSamples = 0
			for value in algorithmOutput.loc[normalIndex]:
				if value == False:
					self.tn += 1
					consecutiveAnomalousSamples = 0
				if value == True:
					consecutiveAnomalousSamples += 1

					if consecutiveAnomalousSamples == KT:
						self.fp += 1
						consecutiveAnomalousSamples=0

			if debug:
				print('Added {} tn'.format(self.tn))
				print('Added {} fp'.format(self.fp))

			"""
			Check anomalous interval
			"""

			consecutiveAnomalousSamples = 0
			detected = False
			for value in algorithmOutput.loc[anomalousIndex]:

				if value == True:
					consecutiveAnomalousSamples += 1
					if consecutiveAnomalousSamples == KT:
						detected = True

				if value == False:
					consecutiveAnomalousSamples = 0

			if detected == True:
				self.tp += 1
			else:
				self.fn += 1

		try:
			self.Precision = float(self.tp)/(float(self.tp) + float(self.fp))
		except:
			if self.fp>0:
				self.Precision = 0
			else:
				self.Precision = np.nan
		try:	
			self.Recall = float(self.tp)/(float(self.tp) + float(self.fn))
		except:
			self.Recall = np.nan

		try:
			self.Accuracy = (float(self.tp) + float(self.tn))/(float(self.tp) + float(self.tn) + float(self.fp) + float(self.fn))
		except:
			self.Accuracy = np.nan

		try: 
			self.Fscore = (1+0.5*0.5) * (self.Precision * self.Recall) / (0.5*0.5*self.Precision + self.Recall)
		except:
			self.Fscore = np.nan

		try:
			self.TNR = self.tn/(self.tn+self.fp)
		except:
			self.TNR = np.nan

		try:
			self.NPV = self.tn/(self.tn+self.fn)
		except:
			self.NPV = np.nan

		try:
			self.FNR = self.fn/(self.fn+self.tp)
		except:
			self.FNR = np.nan

		try:
			self.FDR = self.fp/(self.fp+self.tp)
		except:
			self.FDR = np.nan

		try:
			self.FOR = self.fn/(self.fn+self.tn)
		except:
			self.FOR = np.nan

		try:
			self.TS = self.tp/(self.tp+self.fn+self.fp)
		except:
			self.TS = np.nan

		try:
			self.PT = (np.sqrt(self.Recall*(-self.TNR+1))+self.TNR-1)/(self.Recall+self.TNR-1)
		except:
			self.PT = np.nan

		try:
			self.FPR = self.fp/(self.fp+self.tn)
		except:
			self.FPR = np.nan

		try:
			self.MCC = (self.tp * self.tn - self.fp*self.fn)/(np.sqrt((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn)))
		except:
			self.MCC = np.nan

		try:
			self.MK = self.Precision + self.NPV - 1
		except:
			sefl.MK = np.nan

		try:
			self.BM = self.Recall+self.TNR - 1
		except:
			self.BM = np.nan

		try:
			self.FM = np.sqrt(self.Precision * self.Recall)
		except:
			self.FM = np.nan