# -*- coding: utf-8 -*-
# @Author: andrian
# @Date:   2019-05-06 14:19:39
# @Last Modified by:   Andrian Putina
# @Last Modified time: 2019-11-17 23:11:38

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class Scaler(object):
	"""docstring for Scaler"""
	def __init__(self, scale_type='StandardScaler'):
		super(Scaler, self).__init__()

		if scale_type == 'StandardScaler':
			self.scaler = StandardScaler()

	def fit(self,df):
		self.scaler.fit(df)

	def normalize(self, df):
		"""
		Dataset normalization
		"""
		self.scaler.fit(df)
		return pd.DataFrame(self.scaler.transform(df),
							columns=df.columns)