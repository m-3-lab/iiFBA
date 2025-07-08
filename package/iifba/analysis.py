import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from .utils import input_validation
from .config import GROWTH_MIN_OBJ

def init_iifba(models, media, iterations, m_vals=[1,1]):
    """Initalize storage objects (DataFrames) for iiFBA.
    This function initializes the environment fluxes and original fluxes DataFrames, 
    thereby setting up the structure for the iiFBA analysis.

    Args:
        models (List type of cobra.Model ): 
            List of all COBRApy models to be used for studying interactions in iiFBA
        media (Dict): 
            Dictionary of media with exchange reactions as keys and their fluxes as values.
        iterations (Numeric): 
            Number of iterations to run iiFBA. If float is provided, it will be rounded down 
            to the nearest integer.
        m_vals (List type of Numeric, optional): 
            List of two integers representing the number of sampling runs (starting points)
            and the number of samples taken per sample run/starting point. This is only used
            when sampling is used, otherwise default should be used. If integers are not provided,
            they will be rounded down to the nearest integers.
            Defaults to [1,1].

    Returns:
        env_f: pandas.DataFrame
            Dataframe containing the overall environment fluxes for each iteration and run.
            Multi-indexed by iteration and run, and is consistent across all models.
        org_F: pandas.DataFrame
            Multi-indexed DataFrame containing the fluxes for each model, iteration, and run.
    """
    # get list of all unique rxns and exchanges
    org_exs = set()
    org_rxns = set()
    for model in models:
        exs_set = set(model.exchanges.list_attr("id"))
        org_exs = org_exs | exs_set # exchanges
        rxns_set = set(model.reactions.list_attr("id"))
        org_rxns = org_rxns | rxns_set # reactions

    # initialize env
    rows = (iterations) * m_vals[0] * m_vals[1] + 1 # add one iteration for final env
    cols = len(org_exs)
    env_f = np.zeros((rows, cols))
    env0_masks = [np.array(list(org_exs)) == rxn_id for rxn_id in list(media.keys()) ]
    for flux_idx, flux in enumerate(list(media.values())):
        env_f[0][env0_masks[flux_idx]] = flux
    
    #set columns for multi-indexing
    iters_col = np.repeat(np.arange(1, iterations+1), m_vals[0] * m_vals[1]) 
    run_col = np.tile(np.arange(m_vals[0] * m_vals[1]), iterations)
    iters_col = np.insert(iters_col, 0, 0) # add 0th iteration
    run_col = np.insert(run_col, 0, 0) # add 0th run 
    multi_idx = [iters_col , run_col]
    env_f = pd.DataFrame(env_f, columns=list(org_exs), index=multi_idx) # convert to interprettable df
    env_f.index.names = ["Iteration", "Run"]

    # initialize org_fluxes
    rows = iterations * m_vals[0] * m_vals[1] * len(models)
    cols = len(org_rxns)
    org_F = np.zeros((rows, cols)) # pfba will drop run column
    
    # create unique multi-index for 
    models_col = np.tile(np.arange(len(models)), iterations * m_vals[0] * m_vals[1]) 
    iters_col = np.repeat(np.arange(iterations), m_vals[0] * m_vals[1] * len(models)) 
    run_col = np.tile(np.repeat(np.arange(m_vals[0] * m_vals[1]), len(models)), iterations) 
    multi_idx = [models_col, iters_col , run_col]
    org_F = pd.DataFrame(org_F, columns=list(org_rxns), index=multi_idx)	# convert to interprettable df
    org_F.index.names = ["model", "Iteration", "Run"]
    
    return env_f, org_F

def set_env(model, env_f, iter, run, abundance):
    """Function to set the exhcange reachtions of a model to match the environment fluxes
    for a given iteration and run. This is mainly provided to ensure a cleaner wrapper function.

    Args:
        model (cobra.Model):
            The COBRApy model to set the exchange reactions for.
        env_f (pandas.DataFrame): 
            DataFrame containing the environment fluxes for each iteration and run.
        iter (int): 
            Integer representing the current iteration.
        run (int): 
            Integer representing the current run.

    Returns:
        model (cobra.Model): 
            The updated COBRApy model with exchange reactions set to the environment fluxes.
            Ready for running optimization or sampling for iiFBA analysis.
    """
    for ex in model.exchanges:
        ex.lower_bound = (1/abundance) * env_f.loc[iter, run][ex.id]

    return model


def run_pfba(model, model_idx, iter, org_F, rel_abund):
    """General function to run parsimonious FBA (pFBA) on a model and store the results.
    This function runs pFBA on a given model, checks if the solution is above a minimum growth objective,
    and stores the resulting fluxes in the provided DataFrame.

    Args:
        model (cobra.Model): 
            The COBRApy model to run pFBA on.
        model_idx (int): 
            Integer representing the index of the model in the list of models. Used for 
            indexing in the DataFrame.
        iter (int): 
            Integer representing the current iteration of iiFBA.
        org_F (pandas.DataFrame): 
            DataFrame to store the fluxes from pFBA. It is multi-indexed by model index, 
            iteration, and run.
        rel_abund (float):
            Float value between 0 and 1, representing the relative abundance of the bacteria for the 
            given model.

    Returns:
        org_F (pandas.DataFrame): 
            Updated DataFrame to store the fluxes from iteration of pFBA. It is multi-indexed 
            by model index, iteration, and run (run is 0 for all pFBA).
    """
    # run pFBA
    sol1 = model.slim_optimize()
    if sol1 > GROWTH_MIN_OBJ:
        sol = cb.flux_analysis.parsimonious.pfba(model)
        
        org_F.loc[(model_idx, iter, 0), list(sol.fluxes.index)] = rel_abund * sol.fluxes.values
    # do nothing otherwise - already initiated as zeros!

    return org_F

def run_sampling(model, model_idx, iter, org_F, rel_abund, m_vals, rep_idx, obj_percent):
    """Funtion to run flux sampling on a model and store the results.
        
    Args:
        model (cobra.Model): 
            The COBRApy model to run pFBA on.
        model_idx (int): 
            Integer representing the index of the model in the list of models. Used for 
            indexing in the DataFrame.
        iter (int): 
            Integer representing the current iteration of iiFBA.
        org_F (pandas.DataFrame): 
            DataFrame to store the fluxes from pFBA. It is multi-indexed by model index, 
            iteration, and run.
        rel_abund (float):
            Float value between 0 and 1, representing the relative abundance of the bacteria for the 
            given model.
        m_vals (list type of ints, length 2): 
            List of two integers representing the number of sampling runs (starting points)
            and the number of samples taken per sample run/starting point. If integers are not provided,
            they will be rounded down to the nearest integers.
        rep_idx (int): 
            Integer representing the index of the repetition for sampling. Used for calculating 
            correct indexing in the DataFrame.
        obj_percent (float): 
            Percentage as a float of the objective value to set as the minimum objective value 
            for sampling. obj_percent should be between 0 and 1. If not between 0 and 1, it 
            will default to 0.9.

    Returns:
        org_F (pandas.DataFrame): 
            Updated DataFrame to store the fluxes from iteration of pFBA. It is multi-indexed 
            by model index, iteration, and run.
    """
    # ensure sample space is constrained above a certain objective value
    min_obj = model.slim_optimize() * obj_percent
    
    # set obj to be above min_obj
    obj_rxn = [rxn.id for rxn in linear_reaction_coefficients(model).keys()][0]
    model.reactions.get_by_id(obj_rxn).lower_bound = min_obj

    # run flux sampling
    if iter == 0:
        sample_ct = m_vals[0] * m_vals[1]
    else:
        sample_ct = m_vals[1]
    sol = cb.sampling.sample(model, sample_ct)
    
    # standardize and save output
    arrays = [[model_idx] * sample_ct, [iter] * sample_ct, list(sol.index + rep_idx * sample_ct)]
    tuples = list(zip(*arrays))
    multi_idx = pd.MultiIndex.from_tuples(tuples, names=['model', 'Iteration', 'Run'])
    sol.index = multi_idx
    
    org_F.loc[sol.index, sol.columns] = rel_abund * sol

    return org_F

    
def update_pfba_env(env_f, org_F, rel_abund, iter):
    """Function to update the environment fluxes based on the results of pFBA.
    This function calculates the new environment fluxes by taking into account the
    contributions of each organism's fluxes, weighted by their relative abundances.

    Args:
        env_f (pandas.DataFrame): 
            DataFrame containing the environment fluxes for each iteration and run.
        org_F (pandas.DataFrame): 
            DataFrame to store the fluxes from pFBA. It is multi-indexed by model index, 
            iteration, and run.
        rel_abund (np.ndarray of floats): 
            Relative abundances of the organisms in the environment. This is used to weight the
            contributions of each organism's fluxes to the environment fluxes. The sum of all values
            in rel_abund should be 1, representing the relative proportions of each organism. If not 
            between 0 and 1, it will be normalized to sum to 1. If None, it will default to 1/n where
            n is the number of models.
        iter (int): 
            Integer representing the current iteration of iiFBA.

    Returns:
        env_f (pandas.DataFrame): 
            DataFrame containing the updated environment fluxes for each iteration and run.
    """
    #pull iter info
    env_tmp = env_f.loc[iter, 0][:].to_numpy()
    run_exs = org_F.loc[:, iter, 0][env_f.columns].to_numpy()
    
    # update rate
    update_rate = np.ones_like(rel_abund) # rel_abund


    # run update
    flux_sums = update_rate.T @run_exs
    env_f.loc[iter+1, 0] = env_tmp - flux_sums
    
    return env_f



def update_sampling_env(env_f, org_F, rel_abund, iter, m_vals, Mi, rep_idx):
    """Function to update the environment fluxes based on the results of flux sampling.
    This function calculates the new environment fluxes by taking into account the
    contributions of each organism's fluxes, weighted by their relative abundances.

    Args:
        env_f (pandas.DataFrame): 
            DataFrame containing the environment fluxes for each iteration and run.
        org_F (pandas.DataFrame): 
            DataFrame to store the fluxes from Flux Sampling. It is multi-indexed by model index, 
            iteration, and run.
        rel_abund (np.ndarray of floats): 
            Relative abundances of the organisms in the environment. This is used to weight the
            contributions of each organism's fluxes to the environment fluxes. The sum of all values
            in rel_abund should be 1, representing the relative proportions of each organism. If not 
            between 0 and 1, it will be normalized to sum to 1. If None, it will default to 1/n where
            n is the number of models.
        iter (int): 
            Integer representing the current iteration of iiFBA.
        m_vals (list type of ints, length 2): 
            List of two integers representing the number of sampling runs (starting points)
            and the number of samples taken per sample run/starting point. If integers are not provided,
            they will be rounded down to the nearest integers.
        Mi (int): 
            Integer representing the index of the sampling run (starting point) for this iteration.
            Used for indexing in the DataFrame.
        rep_idx (int): 
            Integer representing the index of the repetition for sampling. Used for calculating 
            correct indexing in the DataFrame.

    Returns:
        env_f (pandas.DataFrame): 
            DataFrame containing the updated environment fluxes for each iteration and run.
    """

    sample_ct = m_vals[0] * m_vals[1] if iter == 0 else m_vals[1]
    for sample_idx in range(sample_ct):
        #pull run info
        env_tmp = env_f.loc[iter, Mi][:].to_numpy()
        run_exs = org_F.loc[:, iter, Mi][env_f.columns].to_numpy()

        # run update
        flux_sums = (run_exs.T @ rel_abund).flatten()
        env_f.loc[iter+1, sample_idx+ m_vals[1]*rep_idx] = env_tmp - flux_sums

    return env_f

def iipfba(models, media, rel_abund="Equal",
           iters=10):
    """Wrapper function for running iiFBA with parsimonious FBA (pFBA). This function initializes 
    the environment and organism fluxes, sets the exchange reactions for each model, runs pFBA,
    and updates the environment fluxes based on the results of pFBA. It returns the
    updated environment fluxes & organism fluxes.

    Args:
        models (List type of cobra.Model ): 
            List of all COBRApy models to be used for studying interactions in iiFBA
        media (Dict): 
            Dictionary of media with exchange reactions as keys and their fluxes as values.
        rel_abund (np.ndarray of floats): 
            Relative abundances of the organisms in the environment. This is used to weight the
            contributions of each organism's fluxes to the environment fluxes. The sum of all values
            in rel_abund should be 1, representing the relative proportions of each organism. If not 
            between 0 and 1, it will be normalized to sum to 1. If None, it will default to 1/n where
            n is the number of models.
        iters (int, optional): 
            Integer value for number of iterations to run. If value is less than 1, iters will default 
            to 1. Decimals will be rounded down to nearest integer. Defaults to 10.

    Returns:
        env_f (pandas.DataFrame): 
            DataFrame containing the environment fluxes for each iteration.
        org_F (pandas.DataFrame): 
            DataFrame to store the fluxes from pFBA. It is multi-indexed by model index &  
            iteration.
    """
    models, media, iters, rel_abund, _, _ = input_validation(models, media, iters, rel_abund)

    env_fluxes, org_fluxes = init_iifba(models, media, iters)

    for iter in range(iters):
        print("Iteration:", iter)

        for org_idx, org_model in enumerate(models):
            with org_model as model:
                # set exchanges
                model = set_env(model, env_fluxes, iter, 0, rel_abund[org_idx]) # only 0 runs

                # run optim
                org_fluxes = run_pfba(model, org_idx, iter, org_fluxes, rel_abund[org_idx])
                
        # update fluxes
        env_fluxes = update_pfba_env(env_fluxes, org_fluxes, rel_abund, iter)

    # pfba has no use for Run index
    env_fluxes = env_fluxes.droplevel("Run")
    org_fluxes =org_fluxes.droplevel("Run")

    return env_fluxes, org_fluxes

def iisampling(models, media, rel_abund, iters=10, m_vals=[1,1], objective_percent= 0.9):
    """Wrapper function for running iiFBA with flux sampling. This function initializes the environment 
    and organism fluxes, sets the exchange reactions for each model, runs flux sampling, and updates the environment fluxes
    based on the results of the flux sampling. It returns the updated environment fluxes, organism
    fluxes, and a matrix of sampling runs (starting points) for each iteration. 

    Args:
        models (List type of cobra.Model ): 
            List of all COBRApy models to be used for studying interactions in iiFBA
        media (Dict): 
            Dictionary of media with exchange reactions as keys and their fluxes as values.
        rel_abund (np.ndarray of floats): 
            Relative abundances of the organisms in the environment. This is used to weight the
            contributions of each organism's fluxes to the environment fluxes. The sum of all values
            in rel_abund should be 1, representing the relative proportions of each organism. If not 
            between 0 and 1, it will be normalized to sum to 1. If None, it will default to 1/n where
            n is the number of models.
        iters (int, optional): 
            Integer value for number of iterations to run. If value is less than 1, iters will default 
            to 1. Decimals will be rounded down to nearest integer. Defaults to 10.
        m_vals (list type of ints, length 2): 
            List of two positive integers representing the number of sampling runs (starting points)
            and the number of samples taken per sample run/starting point. If decimals are provided,
            they will be rounded down to the nearest integers. Negative integers will defaul to 1.
            Defaults to [1,1].
        objective_percent (float, optional): 
            Percentage as a float of the objective value to set as the minimum objective value 
            for sampling. obj_percent should be between 0 and 1. If not between 0 and 1, it 
            will default to 0.9.

    Returns:
        env_f (pandas.DataFrame): 
            DataFrame containing the environment fluxes for each iteration and run.
        org_F (pandas.DataFrame): 
            DataFrame to store the fluxes from flux sampling. It is multi-indexed by model index, 
            iteration, and run.
        M (np.ndarray of ints): 
            Matrix of integers mapping the sampling runs (starting points) for each iteration.
            Each column represents a different iteration, and each row represents a different sampling run.
            The values in the matrix are the indices of the sampling runs (starting points) for that
            iteration. The first column (0th index) is the initial sampling run (starting point) for the first iteration.
    """
    models, media, iters, rel_abund, m_vals, objective_percent = input_validation(models, media, iters, rel_abund, m_vals, objective_percent)

    # mapping of what flux sampling to iterate
    M = np.zeros([m_vals[0],iters],dtype=int) #randomly pre-assign sampling initial point matrix
    for i in range(1, iters):
        Mcol = np.sort(np.random.choice(m_vals[0]*m_vals[1],m_vals[0],replace=False))
        M[:,i]=Mcol

    # initialize env and org fluxes
    env_fluxes, org_fluxes = init_iifba(models, media, iters, m_vals)

    for iter in range(iters):
        print("Iteration:", iter)

        # number of times to re-sample per iteration
        repeat_ct = 1 if iter == 0 else m_vals[0] 
        for rep_idx in range(repeat_ct):
            Mi = M[rep_idx, iter]

            # samples taken
            samples = m_vals[0] * m_vals[1] if iter == 0 else m_vals[1]
            for org_idx, org_model in enumerate(models):
                with org_model as model:
                    # set exchanges
                    model = set_env(model, env_fluxes, iter, Mi)

                    # run optim
                    org_fluxes = run_sampling(model, org_idx, iter, org_fluxes, rel_abund[org_idx], m_vals, rep_idx=rep_idx, obj_percent=objective_percent)
                
        # update fluxes
        env_fluxes = update_sampling_env(env_fluxes, org_fluxes, rel_abund, iter, m_vals, Mi, rep_idx)


    return env_fluxes, org_fluxes, M

def aux_analysis(model):
    """Calculate how well a model grows without metabolites.

    Args:
        model (cobra.Model): 
            The COBRApy model to analyze for essential metabolites.

    Returns:
        essentials (np.ndarray): 
            Array of essential metabolite ids that, when removed, cause a greater than 
            90% reduction in growth.These metabolites are considered essential for the 
            model's growth.
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
    if base_solution < GROWTH_MIN_OBJ: # print a warning if the model didn't grow with all nutrients
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