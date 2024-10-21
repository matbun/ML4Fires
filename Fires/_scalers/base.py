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

import torch
import xarray as xr
from typing import List
from Fires._utilities.decorators import export

@export
class Scaler():
	"""
	Base class for data scalers.

	Attributes:
		features (List[str]):
			List of feature names to be scaled.
		dtype (torch.dtype):
			Data type for scaling operations (default: torch.float32).

	Methods:
		transform(tensor):
			Abstract method for scaling input data.
		inverse_transform(tensor):
			Abstract method for reversing scaling.
	"""
	def __init__(self, features: List[str], dtype=torch.float32) -> None:
		self.features = features
		self.dtype = dtype

	def transform(self, tensor: torch.Tensor):
		raise NotImplementedError

	def inverse_transform(self, tensor: torch.Tensor):
		raise NotImplementedError