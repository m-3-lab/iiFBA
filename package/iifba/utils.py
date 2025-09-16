import matplotlib.pyplot as plt
import cobra as cb
from cobra.util.solver import linear_reaction_coefficients
from importlib.resources import files
import numpy as np
import pandas as pd

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

def plot_sampling(comm, model_idx=None, id=None):
	# set rxn IDs
	if id is None and comm.id is None:
		comm.id = [[rxn.id for rxn in linear_reaction_coefficients(comm.models[model_idx]).keys()][0] for model_idx in range(comm.size)]
	elif id is not None:
		if id in comm.org_fluxes.columns:
			comm.id = [id] * comm.size
		else:
			comm.id = [[rxn.id for rxn in linear_reaction_coefficients(comm.models[model_idx]).keys()][0] for model_idx in range(comm.size)]

	# set model_idx
	if model_idx is None or model_idx >= comm.size:
		model_idx = range(comm.size)
	elif model_idx is not None:
		if not isinstance(model_idx, (list, int)):
			raise ValueError("model_idx must be a list or an integer.")
		if isinstance(model_idx, int):
			model_idx = [model_idx]
		if isinstance(model_idx, list):
			model_idx = [int(idx) for idx in model_idx if idx < comm.size and isinstance(idx, int)]

	# plot
	fig, ax = plt.subplots(1, 1, figsize=(10, 6))
	colors = plt.get_cmap("tab10").colors
	ids = [comm.models[model_idx].id for model_idx in range(comm.size)]
	color_map = {name: colors[i % len(colors)] for i, name in enumerate(ids)}
	sample_ct = comm.m_vals[0] * comm.m_vals[1]
	for org_idx in model_idx:
		print(sample_ct, len(comm.cumulative_fluxes.loc[org_idx, 0, :][comm.id[org_idx]].to_numpy().flatten()))
		ax.scatter(np.ones(sample_ct), comm.cumulative_fluxes.loc[org_idx, 0, :][comm.id[org_idx]].to_numpy().flatten(), label=f"{comm.models[org_idx].id}", color=color_map[ids[org_idx]], alpha=2/(sample_ct))
		for iter in range(2, comm.iters+1):
			ax.scatter(np.full(sample_ct, iter), comm.cumulative_fluxes.loc[org_idx, iter-1, :][comm.id[org_idx]].to_numpy().flatten(), color=color_map[ids[org_idx]], alpha=2/(sample_ct))



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
		["sim1.json"],
		["sim2.json"],
		["sim3_0.json", "sim3_1.json"],
		["sim4_0.json", "sim4_1.json"],
		["sim5_0.json", "sim5_1.json"],
		["sim6_0.json", "sim6_1.json"],
		["sim7_0.json", "sim7_1.json"]
	]

	situation_media = None
	if number in [1,2,4,6,7]: # A only in media 
		situation_media = {"Ex_A": -10}
	elif number in [3]:
		situation_media = {"Ex_A": -10, "Ex_B": -10}
	elif number in [5]:
		situation_media = {"Ex_A": -10, "Ex_C": -10}
	
	models = []
	for file_name in situation_models[number -1]:
		model_path = files("iifba").joinpath("Simple_Models", file_name)
		models.append(cb.io.load_json_model(str(model_path)))
	
	return models, situation_media

def find_min_medium(community=None, models=None, min_growth=None):
	"""result = {k: max(dict1.get(k, float('-inf')), dict2.get(k, float('-inf')))
          for k in set(dict1) | set(dict2)}"""
	
	if community is not None:
		if isinstance(community.media, (float, int)):
			min_growth = community.media
		else:
			min_growth = 0.1
		
		models = community.models
	else:
		min_growth = min_growth if min_growth is not None else 0.1

	min_medium = []
	for model in models:
		model_min_med = cb.medium.minimal_medium(model, min_growth).to_dict()
		min_medium.append(pd.Series(model_min_med))

	min_medium = pd.concat(min_medium, axis=1).fillna(0)
	min_medium = (- 1* min_medium.max(axis=1)).to_dict() # convert to uptake and dict

	return min_medium


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

def check_media(community):
	"""None, complete, [min, 0.10], dict"""

	# None or "complete" == Set all exchanges to -1000
	community.media = "complete" if community.media is None else community.media
	if isinstance(community.media, str):
		if community.media.lower() == "complete":
			community.media = dict(zip(community.org_exs, np.full(len(community.org_exs), -1000)))
		else:
			raise ValueError("Media must be None, 'complete', float, or a dict with reaction IDs as keys and flux values as values.")
	
	if isinstance(community.media, (int, float)):
		community.media = find_min_medium(community)
	elif not isinstance(community.media, (dict, str)):
		raise ValueError("Media must be None, 'complete', float, or a dict with reaction IDs as keys and flux values as values.")

	for rxn_id, flux in community.media.items():
		if not isinstance(rxn_id, str):
			raise ValueError(f"Reaction ID {rxn_id} must be a string.")
		if not isinstance(flux, (int, float)):
			raise ValueError(f"Flux value for reaction {rxn_id} must be a number.")

	return community.media

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

def check_method(method):
	if method is None:
		method = "pfba"
	elif not isinstance(method, str):
		raise ValueError("Method must be a string, either 'pfba' or 'fba'.")
	else:
		if isinstance(method, str):
			if method.lower() == "pfba":
				method = "pfba"
			elif method.lower() == "fba":
				method = "fba"
			else:
				raise ValueError("method must be either 'pfba' or 'fba'.")

	return method

def check_sampling_inputs(community, m_vals):
	if m_vals is None:
		m_vals = [1, 1]
	elif not isinstance(m_vals, (list, np.ndarray)):
		raise ValueError("m_vals must be a list or numpy array.")
	else:
		m_vals = np.array(m_vals, dtype=int)

	if m_vals.ndim != 1 or m_vals.shape[0] != 2:
		raise ValueError("m_vals must be a 1D array of length 2.")

	if np.any(m_vals < 1):
		m_vals[m_vals < 1] = 1

	# convergence: none, logistic: m1*m2>10,000 & iters >2, ks_2: iters>2, energy: iters>3
	if not isinstance(community.sampling_convergence, dict) and community.sampling_convergence is not None:
		raise ValueError("Convergence must be a dictionary.")

	if community.sampling_convergence is None:
		return m_vals, community.sampling_convergence, community.iters 

	if "logistic" not in list(community.sampling_convergence.keys()) and "ks" not in list(community.sampling_convergence.keys()) and "energy" not in list(community.sampling_convergence.keys()):
		raise ValueError("Convergence must be one of: None, 'logistic', 'ks', 'energy'.")

	if "logistic" in community.sampling_convergence:
		if community.iters < 2:
			community.iters = 2
			print("Iterations set to:", community.iters)

		if not isinstance(community.sampling_convergence["logistic"], float):
			raise ValueError("Logistic convergence must be a float.")
		elif community.sampling_convergence["logistic"] < 0 or community.sampling_convergence["logistic"] > 1:
			raise ValueError("Logistic convergence must be a float between 0 and 1.")
		else:
			community.sampling_convergence = {"logistic": float(community.sampling_convergence["logistic"])}

		if m_vals[0] * m_vals[1] < int(7.5 * len(community.org_rxns)):
			raise ValueError(f"Insufficient samples: Logistic convergence requires m_vals[0] * m_vals[1] > 7.5 * n_reactions ({len(community.org_rxns)}).")

	elif "ks" in community.sampling_convergence:
		if community.iters < 2:
			community.iters = 2
			print("Iterations set to:", community.iters)

		if not isinstance(community.sampling_convergence["ks"], float):
			raise ValueError("KS convergence must be a float.")
		elif community.sampling_convergence["ks"] < 0 or community.sampling_convergence["ks"] > 1:
			raise ValueError("KS convergence must be a float between 0 and 1.")
		else:
			community.sampling_convergence = {"ks": float(community.sampling_convergence["ks"])}

	elif "energy" in community.sampling_convergence:
		if community.iters < 3:
			community.iters = 3
			print("Iterations set to:", community.iters)

		if not isinstance(community.sampling_convergence["energy"], float):
			raise ValueError("Energy convergence must be a float.")
		elif community.sampling_convergence["energy"] < 0 or community.sampling_convergence["energy"] > 1:
			raise ValueError("Energy convergence must be a float between 0 and 1.")
		else:
			community.sampling_convergence = {"energy": float(community.sampling_convergence["energy"])}

	return m_vals, community.sampling_convergence, community.iters

