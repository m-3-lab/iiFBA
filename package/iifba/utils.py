import matplotlib.pyplot as plt

# create function for visuals
def iifba_vis():
    return

def store(path, type="pkl"):
    return 
def load_example_models():
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


def input_validation(models=None, media=None, iters=None, flow=None, 
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
	
	if flow is not None:
		if not isinstance(flow, float):
			flow = float(flow)
		if flow < 0 or flow > 1:
			flow = 0.5
		print("Flow set to:", flow)
	
	if rel_abund != None:
		if not isinstance(rel_abund, np.ndarray):
			rel_abund = np.array(rel_abund)
		if rel_abund.ndim != 1:
			raise ValueError("Relative abundances must be a 1D array.")
		if np.any(rel_abund < 0) or np.sum(rel_abund) == 0:
			raise ValueError("Relative abundances must be non-negative and sum to a positive value.")
		if rel_abund.sum() != 1:
			rel_abund = rel_abund / rel_abund.sum()
		if rel_abund is "Equal":
			rel_abund = np.ones(len(models)) / len(models)
		print("Relative abundances set to:", rel_abund)

	if m_vals is not None:
		if not isinstance(m_vals, np.ndarray) or len(m_vals) != 2:
			raise ValueError("m_vals must be a list of two integers.")
		if sum(m_vals > 0) != 2:
			m_vals[m_vals <=0] = 1
		m_vals = np.array(m_vals, dtype=int)
		print("m_vals set to:", m_vals)
	
	if obj_percent is not None:
		if not isinstance(obj_percent, float):
			obj_percent = float(obj_percent)
		if obj_percent < 0 or obj_percent > 1:
			obj_percent = 0.9
		print("Objective percent set to:", obj_percent)
		
	return models, media, iters, flow, rel_abund, m_vals, obj_percent
