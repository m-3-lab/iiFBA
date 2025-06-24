import matplotlib.pyplot as plt

# create function for visuals
def iifba_vis():
    return

def store(path, type="pkl"):
    return 

def input_validation(models=None, media=None, iters=None, flow=None, 
					 rel_abund="None", m_vals=None, obj_percent=None):
	if models is not None or not isinstance(models, list):
		raise ValueError("models must be a list of cobra.Model objects.")
	
	if media is not None or not isinstance(media, dict):
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
	
	if rel_abund != "None":
		if not isinstance(rel_abund, np.ndarray):
			rel_abund = np.array(rel_abund)
		if rel_abund.ndim != 1:
			raise ValueError("Relative abundances must be a 1D array.")
		if np.any(rel_abund < 0) or np.sum(rel_abund) == 0:
			raise ValueError("Relative abundances must be non-negative and sum to a positive value.")
		if rel_abund.sum() != 1:
			rel_abund = rel_abund / rel_abund.sum()
		if rel_abund is None:
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
