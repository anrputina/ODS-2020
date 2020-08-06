# -*- coding: utf-8 -*-
# @Author: andrian
# @Date:   2019-04-02 14:15:11
# @Last Modified by:   andrian
# @Last Modified time: 2019-07-15 16:14:46

import sys
from groundTruth import GroundTruth

class Statistics(object):
	"""docstring for Statistics"""
	def __init__(self, datasetName, configurationFilePath):
		super(Statistics, self).__init__()
		
		self.datasetName = datasetName

		if self.datasetName == 'BGP_testbed_5':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)
		elif self.datasetName == 'BGP_testbed_6':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)			
		elif self.datasetName == 'BGP_testbed_4':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)
		elif self.datasetName == 'BGP_testbed_2':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)
		elif self.datasetName == 'BGP_testbed_3':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)
		elif self.datasetName == 'BGP_testbed_9':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)
		elif self.datasetName == 'BGP_testbed_10':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)	
		elif self.datasetName == 'BGP_VIRL_8':
			self.gt = GroundTruth(self.datasetName, configurationFilePath)													
		else:
			sys.exit('Please select a dataset available in configuration: statistics')

	def computeStatistics_BGP_TESTBED(self, algorithmOutput):
		pass

	def getScores(self, times, algorithmOutput, KT=3):

		if self.datasetName == 'BGP_testbed_5':
			return self.gt.computeScores(times, algorithmOutput, KT)
		if self.datasetName == 'BGP_testbed_6':
			return self.gt.computeScores(times, algorithmOutput, KT)			
		elif self.datasetName == 'BGP_testbed_4':
			return self.gt.computeScores(times, algorithmOutput, KT)
		elif self.datasetName == 'BGP_testbed_2':
			return self.gt.computeScores(times, algorithmOutput, KT)
		elif self.datasetName == 'BGP_testbed_3':
			return self.gt.computeScores(times, algorithmOutput, KT)
		elif self.datasetName == 'BGP_testbed_9':
			return self.gt.computeScores(times, algorithmOutput, KT)	
		elif self.datasetName == 'BGP_testbed_10':
			return self.gt.computeScores(times, algorithmOutput, KT)
		elif self.datasetName == 'BGP_VRIL_8':
			return self.gt.computeScores(times, algorithmOutput, KT)													
		elif self.datasetName == 'NSL-KDD':
			print('Computing statistics for dataset: NSL-KDD')
		else:
			pass