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

import numpy as np
import torch.nn as nn
import lightning.pytorch as pl

from turtle import forward
from typing import Any, Dict, List, Optional
from timm.layers import to_2tuple

from Fires._utilities.decorators import export
from Fires._utilities.logger_itwinai import SimpleItwinaiLogger, ItwinaiLightningLogger 


@export
class BaseLightningModule(pl.LightningModule):
	"""
	A base class for PyTorch Lightning modules, providing essential training, validation, and testing steps.

	Attributes:
		callback_metrics (dict):
			A dictionary to store metrics during training and validation.

	Methods:
		__init__(*args, **kwargs):
			Initializes the base module, setting up callback metrics.
		training_step(batch, batch_idx):
			Performs a training step, calculating loss and logging metrics.
		validation_step(batch, batch_idx):
			Performs a validation step, calculating loss and logging metrics.
		on_validation_model_eval():
			Sets the model to evaluation mode before validation epoch.
		on_validation_model_train():
			Sets the model to training mode after validation epoch.
		on_test_model_train():
			Sets the model to training mode before test epoch.
		on_test_model_eval():
			Sets the model to evaluation mode before test epoch.
		on_predict_model_eval():
			Sets the model to evaluation mode before predict step.
	"""
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.callback_metrics:Dict[str | Any] = {}
	
	@property
	def itwinai_logger(self) -> Optional[SimpleItwinaiLogger]:
		if hasattr(self.trainer, 'loggers'):
			for logger in self.trainer.loggers:
				if isinstance(logger, ItwinaiLightningLogger):
					return logger.logger
		print("WARNING: itwinai_logger non trovato nei trainer loggers.")
		return None

	
	def training_step(self, batch, batch_idx):
		# get data from the batch
		x, y = batch
		# forward pass
		y_pred = self(x)
		# compute loss
		loss = self.loss(y_pred, y)	
		# define log dictionary
		log_dict = {'train_loss': loss}
		
		# binarize real and predicted data
		y_true_bin = (y > 0).int()
		y_pred_bin = (y_pred > 0).int()

		# flatten tensors
		y_true_flat = y_true_bin.view(-1)
		y_pred_flat = y_pred_bin.view(-1)

		# compute metrics
		for metric in self.metrics:
			metric_name = f'train_{metric.name.lower()}'
			log_dict[metric_name] = metric(y_pred_flat, y_true_flat)
		# log the outputs
		self.callback_metrics = {**self.callback_metrics, **log_dict}

		# Log with itwinai logger all the hyperparameters from training step
		if self.itwinai_logger:
			# Log hyper-parameters
			self.itwinai_logger.save_hyperparameters(self.callback_metrics)

		# return the loss
		return {'loss':loss}

	def validation_step(self, batch, batch_idx):
		# get data from the batch
		x, y = batch
		# forward pass
		y_pred = self(x)
		# compute loss
		loss = self.loss(y_pred, y)
		# define log dictionary
		log_dict = {'val_loss': loss}

		# binarize real and predicted data
		y_true_bin = (y > 0).int()
		y_pred_bin = (y_pred > 0).int()

		# flatten tensors
		y_true_flat = y_true_bin.view(-1)
		y_pred_flat = y_pred_bin.view(-1)

		# compute metrics
		for metric in self.metrics:
			metric_name = f'val_{metric.name.lower()}'
			log_dict[metric_name] = metric(y_pred_flat, y_true_flat)
		# log the outputs
		self.callback_metrics = {**self.callback_metrics, **log_dict}

		# Log with itwinai logger all the hyperparameters from validation step
		if self.itwinai_logger:
			# Log hyper-parameters
			self.itwinai_logger.save_hyperparameters(self.callback_metrics)

		# return the loss
		return {'loss':loss}

	def on_validation_model_eval(self) -> None:
		self.eval()
	def on_validation_model_train(self) -> None:
		self.train()
	def on_test_model_train(self) -> None:
		self.train()
	def on_test_model_eval(self) -> None:
		self.eval()
	def on_predict_model_eval(self) -> None:
		self.eval()


@export
class BaseUnetPlusPlus(BaseLightningModule):
	"""
	A base class for U-Net++ models, inheriting from BaseLightningModule.

	Attributes:
		input_shape (tuple):
			The shape of input images (height, width, channels).
		num_classes (int):
			The number of output classes for segmentation.
		depth (int):
			The depth of the U-Net++ architecture.
		base_filter_dim (int):
			The number of filters in the first layer.
		deep_supervision (bool):
			Whether to use deep supervision.
		model (nn.Module):
			The underlying U-Net++ model (initialized as nn.Identity).

	Methods:
		__init__(input_shape, num_classes, depth, base_filter_dim, deep_supervision, *args, **kwargs):
			Initializes the base U-Net++ model with specified parameters.
		forward(inputs):
			Performs a forward pass through the U-Net++ model.
	"""
	def __init__(self,
			input_shape:tuple=(720, 1440, 8),
			num_classes:int=1,
			depth:int=4,
			base_filter_dim:int=32,
			deep_supervision:bool=False,
			*args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.depth = depth
		self.base_filter_dim = base_filter_dim
		self.deep_supervision = deep_supervision
		self.model = nn.Identity()
	
	def forward(self, inputs) -> Any:
		return self.model(inputs)


@export
class BaseVGG(BaseLightningModule):
	"""
	A base class for VGG-like models, inheriting from BaseLightningModule.

	Attributes:
		channels (List[int]):
			A list of channel numbers for each convolutional block.
		activation (nn.Module):
			The activation function to use after each convolutional block.
		kernel_size (int):
			The size of the convolutional kernels.
		model (nn.Module):
			The underlying VGG-like model (initialized as nn.Identity).

	Methods:
		__init__(channels, activation, kernel_size, *args, **kwargs):
			Initializes the base VGG model with specified parameters.
		forward(inputs):
			Performs a forward pass through the VGG model.
	"""

	def __init__(self, 
			channels: List[int], 
			activation: nn.Module = nn.Identity, 
			kernel_size: int = 3, 
			*args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.channels = channels
		self.activation = activation
		self.kernel_size = kernel_size
		self.model = nn.Identity()

	def forward(self, inputs) -> Any:
		return self.model(inputs)
