import matplotlib.pyplot as plt
import cobra as cb
from importlib.resources import files
import numpy as np
import pandas as pd
import os

def iifba_vis(F, ax=None, 
			  line_lab=None):
	"""Plot the cumulative flux values from iifba results. 

	This function returns a matplotlib Axes object with the cumulative flux,
	which can be customized further or used for overlaying on other plots.
	Change colors or other attributes of the lines with:
	
	lines = ax.get_lines() # Get all lines on the axes
	lines[0].set_color('red') # For example, change the color of the first line

	Args:
		F (1D array-like): 
			Flux values to be plotted. Flux values should be in the 
			form of a 1D array or series, where each value corresponds to
			flux of a reaction at a specific iteration.
		ax (matplotlib.axes.Axes, optional): 
			Matplotlib Axes to plot iifba results. 
			Defaults to None.
		line_lab (str, optional): 
			In case of overlaying plot, used for labeling lines. Defaults to None.

	Returns:
		ax (matplotlib.axes.Axes, optional): 
			Matplotlib Axes to plot iifba results. Returns for saving or
			further customization or overlaying on other plots.
	"""
	if ax is None:
		_, ax = plt.subplots(1,1)
	
	ax.plot(np.cumsum(F), label=line_lab)
	ax.set_ylabel("Cumulative Flux")
	ax.set_xlabel("Iteration")

	if line_lab is not None:
		ax.legend()
	
	return ax

def store(path, type="pkl"):
    return 
def load_example_models():
	"""Load real bacteria models for testing purposes.

	Returns:
		example_EC: cobra.Model
			Escherichia coli str. K-12 substr. MG1655 model.
		example_BT: cobra.Model
			Bacteroides thetaiotaomicron 3731 model.
		ecoli_media_example: dict
			Example glucose minimal media for E. coli, with reaction IDs 
			as keys and flux values as
	"""
	mat_path = files("iifba").joinpath("AGORA2_Models", "Escherichia_coli_str_K_12_substr_MG1655.mat")
	example_EC = cb.io.load_matlab_model(str(mat_path))

	mat_path = files("iifba").joinpath("AGORA2_Models", "Bacteroides_thetaiotaomicron_3731.mat")
	example_BT = cb.io.load_matlab_model(str(mat_path))
	
	#ecoli example glucose minimal media
	min_med_ids_ex = ['EX_glc_D(e)','EX_so4(e)','EX_nh4(e)','EX_no3(e)','EX_pi(e)','EX_cys_L(e)',
				'EX_mn2(e)','EX_cl(e)','EX_ca2(e)','EX_mg2(e)','EX_cu2(e)','EX_cobalt2(e)','EX_fe2(e)','EX_fe3(e)','EX_zn2(e)','EX_k(e)']
	# Define medium uptake flux bounds
	min_med_fluxes_ex = [-10,-100,-100,-100,-100,-100,
						-100,-100,-100,-100,-100,-100,-100,-100,-100,-100]
	ecoli_media_example = dict(zip(min_med_ids_ex, min_med_fluxes_ex))
	
	return example_EC, example_BT, ecoli_media_example

def load_simple_models(number):
	situation_models = [
		["sim_I.json"],
		["sim_II.json"],
		["sim_III_1.json", "sim_III_2.json"],
		["sim_IV_1.json", "sim_IV_2.json"],
		["sim_V_1.json", "sim_V_2.json"],
		["sim_VI_1.json", "sim_VI_2.json"],
		["sim_VII_1.json", "sim_VII_2.json"]
	]

	situation_media = None
	if number in [1, 2, 3,5, 7]: # A only in media 
		situation_media = {"Ex_A": -10}
	elif number in [4]:
		situation_media = {"Ex_A": -10, "Ex_B": -10}
	elif number in [6]:
		situation_media = {"Ex_A": -10, "Ex_C": -10}
	
	models = []
	for file_name in situation_models[number -1]:
		model_path = files("iifba").joinpath("Simple_Models", file_name)
		models.append(cb.io.load_json_model(str(model_path)))
	
	return models, situation_media

def input_validation(models=None, media=None, iters=None, 
					 rel_abund=None, m_vals=None, obj_percent=None):
	if models is not None and not isinstance(models, list):
		raise ValueError("models must be a list of cobra.Model objects.")
	
	if media is not None and not isinstance(media, dict):
		raise ValueError("media must be a dictionary with reaction IDs as keys and flux values as values.")
	
	if iters is not None:
		if not isinstance(iters, int):
			iters = int(iters)
		if iters < 1:
			iters = 1
		print("Iterations set to:", iters)
	
	if rel_abund is not None:
		if isinstance(rel_abund, str):
			rel_abund = np.ones(len(models)) / len(models)
		if not isinstance(rel_abund, np.ndarray):
			rel_abund = np.array(rel_abund)
		if rel_abund.ndim != 1:
			raise ValueError("Relative abundances must be a 1D array.")
		if np.any(rel_abund < 0) or np.sum(rel_abund) == 0:
			raise ValueError("Relative abundances must be non-negative and sum to a positive value.")
		if rel_abund.sum() != 1:
			rel_abund = rel_abund / rel_abund.sum()
		print("Relative abundances set to:", rel_abund)

	if m_vals is not None:
		m_vals = np.array(m_vals, dtype=int)
		if len(m_vals) != 2:
			raise ValueError("m_vals must be a list of two integers.")
		if sum(m_vals > 0) != 2:
			m_vals[m_vals <=0] = 1
		print("m_vals set to:", m_vals)
	
	if obj_percent is not None:
		if not isinstance(obj_percent, float):
			obj_percent = float(obj_percent)
		if obj_percent < 0 or obj_percent > 1:
			obj_percent = 0.9
		print("Objective percent set to:", obj_percent)
		
	return models, media, iters, rel_abund, m_vals, obj_percent

def check_rel_abund(rel_abund, n_models):
	if rel_abund is None:
		rel_abund = np.ones(n_models) / n_models
	elif isinstance(rel_abund, str):
		rel_abund = np.ones(n_models) / n_models
	elif not isinstance(rel_abund, np.ndarray):
		rel_abund = np.array(rel_abund)
	if rel_abund.ndim != 1:
		rel_abund = rel_abund.flatten()
	if rel_abund.shape[0] != n_models:
		raise ValueError(f"Relative abundances must be a 1D array of length {n_models}.")
	if np.any(rel_abund < 0) or np.sum(rel_abund) == 0:
		raise ValueError("Relative abundances must be non-negative and sum to a positive value.")
	if rel_abund.sum() != 1:
		rel_abund = rel_abund / rel_abund.sum()
		print("Relative abundances set to:", rel_abund)

	rel_abund = rel_abund.astype(float).reshape(-1, 1)
	return rel_abund

def check_iters(iters):
	if iters is None:
		iters = 10
	elif not isinstance(iters, int):
		iters = int(iters)
	if iters < 1:
		iters = 1
		print("Iterations set to:", iters)
	
	return iters

def check_media(media):
	if media is None:
		media = {}
	elif not isinstance(media, dict):
		raise ValueError("Media must be a dictionary with reaction IDs as keys and flux values as values.")
	
	for rxn_id, flux in media.items():
		if not isinstance(rxn_id, str):
			raise ValueError(f"Reaction ID {rxn_id} must be a string.")
		if not isinstance(flux, (int, float)):
			raise ValueError(f"Flux value for reaction {rxn_id} must be a number.")
	
	return media

def check_models(models):
	if models is None:
		raise ValueError("Models must be provided as a list of cobra.Model objects or single cobra.Model.")

	elif not isinstance(models, (list, cb.Model)):
		raise ValueError("Models must be provided as a list of cobra.Model objects or single cobra.Model.")
	else:
		if isinstance(models, cb.Model):
			models = [models]

	for model in models:
		if not isinstance(model, cb.Model):
			raise ValueError(f"Model {model} is not a valid cobra.Model object.")
	
	return models

def check_objective(objective):
	if objective is None:
		objective = "pfba"
	elif not isinstance(objective, str):
		raise ValueError("Objective must be a string, either 'pfba' or 'fba'.")
	else:
		if isinstance(objective, str):
			if objective.lower() == "pfba":
				objective = "pfba"
			elif objective.lower() == "fba":
				objective = "fba"
			else:
				raise ValueError("Objective must be either 'pfba' or 'fba'.")

	return objective