import cobra as cb
import numpy as np
import pandas as pd

# Simple non-sampling
def iifba(community_model, media, relative_abundance,
          flow=0.5, solution_type="pFBA", 
          iterations=10,
          m_vals=[1,1], v=False):
    """
    Summary:

    
    Params:
    - community_model: LIST type of Cobra Models (len number of unique bacteria)
    descr.

    - media: pd.DataFrame ()
    descr.

    - relative_abundance: LIST type of FLOAT (len number of unique bacteria)
    relative abundances of each bacteria in community. If sum(relative_abundance) > 1,
    relative abundances will be scaled by the sum. 

    - flow (optional): FLOAT
    default = 0.5
    Input flow rate of new metabolites/exchanges in media

    - solution type (optional): STR
    default = "pFBA"
    Type of optimization for FLux balance
    can be "pFBA", "sampling"

    - iterations (optional): INT
    default = 10
    THe number of interations until completion. Must be greater than 1 iteration.

    - m_vals (optional): LIST type of INT (2,)
    default = [1, 1]
    Number of initial flux points to use in flux sampling and number of runs per 
    iterations. If both values are 1, then simple 1-to-1 iterations are done.

    - v (optional) BOOL
    default = False
    Turn on verbose or turn off

    
    Returns:
    - flux_log: pandas.Dataframe 
    Contains values of all fluxes in exchanges of the community. Dataframe is
    multi-indexed by (iteration, run), run will always be 0 if using pFBA.

    - F: LIST of pandas.Dataframe
    Each index of list corresponds to the model of community_model. Each dataframe 
    contains all the fluxes of the appropriate model. Dataframe is multi-indexed 
    by (iteration, run), run will always be 0 if using pFBA.

    
    """
    # convert all numeric to ints to ensure proper variable useage
    m_vals[0] = int(m_vals[0])
    m_vals[1] = int(m_vals[1])
    iterations = int(iterations)
    if solution_type.lower() == "pfba":
        print("Using Parsimonious FBA")
        m_vals = [1,1]
    elif solution_type.lower() == "sampling":
        print("Using Flux Sampling")
    else:
        print("Defaulting to Using Parsimonious FBA")
        solution_type = "pfba"
    solution_type = solution_type.lower()

    if sum(relative_abundance) >1:
        print("Scaling Abundance") if v else None
        relative_abundance = [r/sum(relative_abundance) for r in relative_abundance]

    print("Initializing Iterations") if v else None
    M = np.zeros((m_vals[0], iterations -1), dtype=int)
    for i in range(iterations-1):
        Mcol = np.sort(np.random.choice(m_vals[0]*m_vals[1],m_vals[0],replace=False))
        M[:,i]=Mcol
        

    # store fluxes of all exchange reactions for the overall model based on media
    print("Initializing Exchanges Logging") if v else None
    arrays = [[0]*m_vals[0]*m_vals[1],list(range(m_vals[0]*m_vals[1]))]
    tuples = list(zip(*arrays))
    multi_idx = pd.MultiIndex.from_tuples(tuples,names=['iteration','run'])

    # extract all exchange reactions
    cols = set()
    for model_idx in range(len(community_model)):
        for model_ex in range(len(community_model[model_idx].exchanges)):
            cols.add(community_model[model_idx].exchanges[model_ex].id)

    # compile initial media conditions
    print("Initializing Environment Logging") if v else None
    flux_log = pd.DataFrame([np.zeros(len(cols))],
                     columns=list(cols),
                     index=multi_idx,dtype=float)
    for media_ex in range(len(media)):
        exid = media.iloc[media_ex]['Reaction']
        ex_flux = media.iloc[media_ex]['LB']
        flux_log.loc[:,exid] = ex_flux
    
    # initialize organism flux dataframes
    F = []  

    # iterations
    print("Running Iterations") if v else None
    for iter in range(iterations):
        print("Iteration:", iter)
        
        if iter == 0:
            # use media for the first time around for all models
            for org_idx in range(len(community_model)):
                print("Organism:", org_idx)
                with community_model[org_idx] as model_iter:
                    # reset exchanges for environment setting
                    print("Reset Exchanges") if v else None
                    for ex in model_iter.exchanges:
                        ex.lower_bound = 0
                        ex.upper_bound = 1000
                    
                    # Set Environment for 0th run (same initial env. for all runs)
                    print("Set Environment") if v else None
                    for env_ex in range(len(flux_log.columns)):
                        ex_lb = flux_log.loc[(0,0)][flux_log.columns[env_ex]] #initial environment is the same for all runs, so use the 0th run
                        if ex_lb != 0:
                            ex_id = flux_log.columns[env_ex]
                            if ex_id in model_iter.exchanges:
                                model_iter.exchanges.get_by_id(ex_id).lower_bound = ex_lb
                    
                    # run optimization with pfba
                    if solution_type == 'pfba':
                        print("Running Optimization") if v else None
                        multi_idx = pd.MultiIndex.from_tuples([(0,0)],names=["iteration","run"])                                       
                        # run pFBA
                        sol1 = model_iter.slim_optimize()
                        if sol1 > 0.001:
                            sol = cb.flux_analysis.parsimonious.pfba(model_iter)
                            # standardize and save output                   
                            df = pd.DataFrame([sol.fluxes],columns=sol.fluxes.index,index=multi_idx)
                            F.append(df)
                        else:
                            # if no growth and cannot use the solution
                            rxnid = []
                            for i in range(len(model_iter.reactions)): 
                                rxnid.append(model_iter.reactions[i].id)
                            df = pd.DataFrame([np.zeros(len(model_iter.reactions))],columns=rxnid,index=multi_idx)
                            F.append(df)
                    
                    # run optimization with flux sampling
                    if solution_type == 'sampling':
                        print("Running Optimization") if v else None
                        # run flux sampling
                        sol = cb.sampling.sample(model_iter, m_vals[0]*m_vals[1])
                        # standardize and save output
                        arrays = [[0]*m_vals[0]*m_vals[1],list(sol.index)]
                        tuples = list(zip(*arrays))
                        multi_idx = pd.MultiIndex.from_tuples(tuples,names=['iteration','run'])
                        sol.index = multi_idx
                        F.append(sol)

            # update f
            for run_idx in range(m_vals[0]*m_vals[1]): 
                print("Updating Fluxes") if v else None
                env_tmp = flux_log.loc[[(iter,0)]].copy(deep=True) #temporary dataframe for base environment from iteration 0,0
                for env_ex in range(len(flux_log.columns)):# for each exchange flux in environment
                    ex_flux_sum = 0
                    ex_flux_id = flux_log.columns[env_ex]
                    #sum total flux of all bacteria in model
                    for org_idx in range(len(community_model)):# for each organism sum up flux * relative abundance
                        if ex_flux_id in community_model[org_idx].exchanges:
                            if F[org_idx].loc[(0,run_idx)][ex_flux_id] != 0:
                                ex_flux_sum += F[org_idx].loc[(0,run_idx)][ex_flux_id] * relative_abundance[org_idx]

                    #iifba update for ex
                    env_tmp.loc[(0,0),ex_flux_id] = (1-flow)*(flux_log.loc[(0,0)][ex_flux_id].item()-ex_flux_sum) + flow*flux_log.loc[(0,0)][ex_flux_id].item() # update flux
                
                #re-index tmp dataframe
                multi_idx = pd.MultiIndex.from_tuples([(1,run_idx)],names=["iteration","run"])
                df_tt = pd.DataFrame([env_tmp.loc[(0,0)]],columns = env_tmp.columns, index = multi_idx)
                flux_log = pd.concat([flux_log,df_tt])
        
        # re-run for other iterations
        else:       
            # if flux sampling, repeat for multiple points
            for m1_idx in range(m_vals[0]):
                M_iter = M[m1_idx, iter-1]

                # run iteration for all bacteria in community
                for org_idx in range(len(community_model)):
                    print('organism:',org_idx)

                    with community_model[org_idx] as model_iter:
                        # reset exchanges for environment setting
                        print("Reset Exchanges") if v else None
                        for ex in model_iter.exchanges:
                            ex.lower_bound = 0
                            ex.upper_bound = 1000
                    
                        # Set Environment
                        print("Set Environment") if v else None
                        for env_ex in range(len(flux_log.columns)):
                            ex_lb = flux_log.loc[(iter,M_iter)][flux_log.columns[env_ex]]
                            if ex_lb != 0:
                                ex_id = flux_log.columns[env_ex]
                                if ex_id in model_iter.exchanges:
                                    model_iter.exchanges.get_by_id(ex_id).lower_bound = ex_lb
                        

                        if solution_type == 'pfba':
                            print("Running Optimization") if v else None
                            multi_idx = pd.MultiIndex.from_tuples([(iter,0)],names=["iteration","run"])                                       
                            # run pFBA
                            sol1 = model_iter.slim_optimize()
                            if sol1 > 0.001:
                                sol = cb.flux_analysis.parsimonious.pfba(model_iter)
                                # standardize and save output                   
                                df = pd.DataFrame([sol.fluxes],columns=sol.fluxes.index,index=multi_idx)
                                F[org_idx] = pd.concat([F[org_idx],df])
                            else:
                                rxnid = []
                                for i in range(len(model_iter.reactions)): 
                                    rxnid.append(model_iter.reactions[i].id)
                                df = pd.DataFrame([np.zeros(len(model_iter.reactions))],columns=rxnid,index=multi_idx)
                                F[org_idx] = pd.concat([F[org_idx],df])

                        if solution_type == 'sampling':
                            print("Running Optimization") if v else None
                            # run flux sampling
                            sol = cb.sampling.sample(model_iter,m_vals[0])
                            # standardize and save output
                            arrays = [[iter]*m_vals[0]*m_vals[1],list(sol.index+m1_idx*m_vals[1])]
                            tuples = list(zip(*arrays))
                            multi_idx = pd.MultiIndex.from_tuples(tuples,names=['iteration','run'])
                            sol.index = multi_idx
                            F[org_idx] = pd.concat([F[org_idx],sol])
            
            # update fluxes
            for m2_idx in range(m_vals[1]):
                print("Updating Fluxes") if v else None
                env_tmp = flux_log.loc[[(iter,M_iter)]].copy(deep=True) #temporary dataframe for base environment from iteration 0,0
                for ex_idx in range(len(flux_log.columns)):# for each exchange flux in environment
                    ex_flux_sum = 0
                    ex_flux_id = flux_log.columns[ex_idx]
                    for org_idx in range(len(community_model)):# for each organism sum up flux * relative abundance
                        if ex_flux_id in community_model[org_idx].exchanges:
                            if F[org_idx].loc[(iter, m2_idx+m1_idx*m_vals[1])][ex_flux_id] != 0:
                                ex_flux_sum += F[org_idx].loc[(iter,m2_idx+m1_idx*m_vals[1])][ex_flux_id] * relative_abundance[org_idx]

                    env_tmp.loc[(iter,M_iter),ex_flux_id] = (1-flow)*(flux_log.loc[(iter,M_iter)][ex_flux_id].item()-ex_flux_sum) + flow*flux_log.loc[(0,0)][ex_flux_id].item() # update flux
                #re-index tmp dataframe
                multi_idx = pd.MultiIndex.from_tuples([(iter+1,m2_idx+m1_idx*m_vals[1])],names=["iteration","run"])
                df_tt = pd.DataFrame([env_tmp.loc[(iter,M_iter)]],columns = env_tmp.columns, index = multi_idx)
                flux_log = pd.concat([flux_log,df_tt])

    return flux_log, F

def aux_analysis(model):
    """Calculate how well a model grows without metabolites.

    Args:
        model (cobra.Model): _description_

    Returns:
        essentials (np.Array): _description_
    """
    exchange = [] # Keep track of exchanges
    relative_growth = [] # Keep track of relative growth rate (how well does the model grow without this metabolite vs with the metabolite)

    # turn on model exchanges
    for ex in model.exchanges:
        ex.lower_bound = -1000
        ex.upper_bound = 1000

    # turn off O2 for base anaerobic growth
    model.exchanges.get_by_id('EX_o2(e)').lower_bound = 0 # turn off oxygen uptake (anaerobic)

    # calculate the base growth rate, with all nutrients available, using FBA
    base_solution = model.slim_optimize() 
    if base_solution < 0.01: # print a warning if the model didn't grow with all nutrients
        print('base solution low')
        
    for ex in model.exchanges: # for each exchange reaction
        ex_id = ex.id
        
        with model as model1: 
            model1.reactions.get_by_id(ex_id).lower_bound = 0  # turn off uptake
            aux_solution = model.slim_optimize()  # recalculate growth rate
            rel_gro = (aux_solution-base_solution)/base_solution # calculate relative growth change
            exchange.append(ex_id) # save exchange id
            relative_growth.append(rel_gro) # save relative growth change

    # Save essential metabolite ids for later (these are the metabolites that when removed cause a greater than 90% reduction in growth)
    essential_inds = np.argwhere(np.array(relative_growth)<-0.9)
    essentials = np.array(exchange)[essential_inds]

    return essentials