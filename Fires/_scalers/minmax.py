# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2024 - CMCC Foundation						
#																			
# Site: 			https://www.cmcc.it										
# CMCC Institute:	IESP (Institute for Earth System Predictions)
# CMCC Division:	ASC (Advanced Scientific Computing)						
# Author:			Emanuele Donno											
# Email:			emanuele.donno@cmcc.it									
# 																			
# Licensed under the Apache License, Version 2.0 (the "License");			
# you may not use this file except in compliance with the License.			
# You may obtain a copy of the License at									
#																			
#				https://www.apache.org/licenses/LICENSE-2.0					
#																			
# Unless required by applicable law or agreed to in writing, software		
# distributed under the License is distributed on an "AS IS" BASIS,			
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	
# See the License for the specific language governing permissions and		
# limitations under the License.											
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch
import xarray as xr
from typing import List
from Fires._utilities.decorators import export
from Fires._scalers.base import Scaler

@export
class MinMaxScaler(Scaler):
	"""
	Scales data to a specified range using minimum and maximum values.

	Attributes:
		min_ds (xr.Dataset):
			xarray Dataset containing minimum values for features.
		max_ds (xr.Dataset):
			xarray Dataset containing maximum values for features.
		coeff_ (torch.Tensor):
			Scaling coefficients derived from min and max values.
		element_ (torch.Tensor):
			Scaling offset derived from min and max values.

	Methods:
		__init__(min_ds, max_ds, features, dtype):
			Initializes the scaler.
		transform(tensor):
			Scales input tensor to the range [0, 1].
		inverse_transform(tensor):
			Reverses the scaling operation.
	"""
	def __init__(self, min_ds, max_ds, features: List[str], dtype=torch.float32) -> None:
		super().__init__(features, dtype)

		if not features:
			raise ValueError("Features list must be non-empty!")
		
		if not min_ds:
			raise ValueError("Data for min values must not be empty!")
		
		if not max_ds:
			raise ValueError("Data for min values must not be empty!")
		
		min_ds = min_ds[features]
		max_ds = max_ds[features]

		coeff = 1 / (max_ds - min_ds)
		element = min_ds * coeff

		self.coeff_ = torch.as_tensor(coeff.to_array().data, dtype=self.dtype).view(8, 1, 1)
		self.element_ = torch.as_tensor(element.to_array().data, dtype=self.dtype).view(8, 1, 1)

	def transform(self, tensor: torch.Tensor):
		scaled_tensor = tensor * self.coeff_ - self.element_
		return scaled_tensor
	
	def inverse_transform(self, tensor: torch.Tensor):
		rescaled_tensor = (tensor + self.element_) / self.coeff_
		return rescaled_tensor