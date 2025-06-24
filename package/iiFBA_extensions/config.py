import cobra as cb

ecoli_example = [cb.io.load_matlab_model("AGORA2_Models/Escherichia_coli_str_K_12_substr_MG1655.mat")]

#ecoli example glucose minimal media
min_med_ids_ex = ['EX_glc_D(e)','EX_so4(e)','EX_nh4(e)','EX_no3(e)','EX_pi(e)','EX_cys_L(e)',
			   'EX_mn2(e)','EX_cl(e)','EX_ca2(e)','EX_mg2(e)','EX_cu2(e)','EX_cobalt2(e)','EX_fe2(e)','EX_fe3(e)','EX_zn2(e)','EX_k(e)']
# Define medium uptake flux bounds
min_med_fluxes_ex = [-10,-100,-100,-100,-100,-100,
					-100,-100,-100,-100,-100,-100,-100,-100,-100,-100]
ecoli_media_example = dict(zip(min_med_ids_ex, min_med_fluxes_ex))

GROWTH_MIN_OBJ = 0.01

