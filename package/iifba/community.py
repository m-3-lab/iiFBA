import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from . import utils
from .config import GROWTH_MIN_OBJ, ROUND
from .summary import CommunitySummary

class Community:
    """
    A class to represent a community of organisms in iiFBA analysis.
    
    Attributes
    ----------
    """

    def __init__(self, models, media, rel_abund=None, iters=10, objective= "pfba", early_stop=True, m_vals=[1,1], v=False):
        # input parameters validation
        self.models = utils.check_models(models)
        self.size = len(models)
        self.media = utils.check_media(media)
        self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        self.iters = utils.check_iters(iters)
        self.m_vals = m_vals
        self.objective = utils.check_objective(objective)
        # get obj rxn ids
        model_obj_rxns = []
        for model in self.models:
            obj_rxn = linear_reaction_coefficients(model).keys()
            model_obj_rxns.extend([rxn.id for rxn in obj_rxn])
        self.objective_rxns = dict(zip(range(self.size), 
                                      model_obj_rxns))

        # iterator parameters
        self.v = v if not v else True  # verbose mode
        self.early_stop = early_stop if early_stop else False

        self.create_variables()

    def create_variables(self):
        """Initialize variables for iiFBA.
        This function sets up the optimization variables for the iiFBA analysis.
        """
        # get list of all unique rxns and exchanges
        self.ex_to_met = {}
        self.exchange_metabolites = []
        self.exchanges = []
        self.org_exs = set()
        self.org_rxns = set()

        # rxns/echanges/boundary mets per model
        for model in self.models:
            exs_set = set(model.exchanges.list_attr("id"))
            self.org_exs = self.org_exs | exs_set # exchanges

            rxns_set = set(model.reactions.list_attr("id"))
            self.org_rxns = self.org_rxns | rxns_set # reactions

            for rxn in model.exchanges:
                mets = list(rxn.metabolites.keys())
                if len(mets) == 1:
                    self.ex_to_met[rxn.id] = mets[0].id if pd.notnull(mets[0].id) else rxn.id
                    self.exchange_metabolites.extend(mets)
                    self.exchanges.append(rxn.id)
        
        # convert to attribute lists
        self.org_exs = list(self.org_exs)
        self.org_rxns = list(self.org_rxns)
        self.exchange_metabolites = list(set(self.exchange_metabolites))
        self.exchanges = list(set(self.exchanges))

        # initialize env
        rows = (self.iters) * self.m_vals[0] * self.m_vals[1] + 1 # add one iteration for final env
        cols = len(self.org_exs)
        self.env_fluxes = np.zeros((rows, cols))
        env0_masks = [np.array(self.org_exs) == rxn_id for rxn_id in list(self.media.keys())]
        for flux_idx, flux in enumerate(list(self.media.values())):
            self.env_fluxes[0][env0_masks[flux_idx]] = -flux

        #set columns for multi-indexing
        iters_col = np.repeat(np.arange(1, self.iters+1), self.m_vals[0] * self.m_vals[1]) 
        run_col = np.tile(np.arange(self.m_vals[0] * self.m_vals[1]), self.iters)
        iters_col = np.insert(iters_col, 0, 0) # add 0th iteration
        run_col = np.insert(run_col, 0, 0) # add 0th run 
        multi_idx = [iters_col , run_col]
        self.env_fluxes = pd.DataFrame(self.env_fluxes, columns=self.org_exs, index=multi_idx) # convert to interprettable df
        self.env_fluxes.index.names = ["Iteration", "Run"]

        # initialize org_fluxes
        rows = self.iters * self.m_vals[0] * self.m_vals[1] * len(self.models)
        cols = len(self.org_rxns)
        self.org_fluxes = np.zeros((rows, cols)) # pfba will drop run column
        
        # create unique multi-index for 
        models_col = np.tile(np.arange(self.size), self.iters * self.m_vals[0] * self.m_vals[1]) 
        iters_col = np.repeat(np.arange(self.iters), self.m_vals[0] * self.m_vals[1] * self.size) 
        run_col = np.tile(np.repeat(np.arange(self.m_vals[0] * self.m_vals[1]), self.size), self.iters) 
        multi_idx = [models_col, iters_col , run_col]
        self.org_fluxes = pd.DataFrame(self.org_fluxes, columns=self.org_rxns, index=multi_idx)	# convert to interprettable df
        self.org_fluxes.index.names = ["Model", "Iteration", "Run"]
        
        return 

    def set_env(self, model_idx, low_bounds):
        """Function to set the exhcange reactions of a model to match the environment fluxes
        for a given iteration and run. This is mainly provided to ensure a cleaner wrapper function.
        """
        for ex in self.models[model_idx].exchanges:
            mask = np.array(self.org_exs) == ex.id
            if mask.any():  # Check if the exchange reaction exists in org_exs
                ex.lower_bound = -low_bounds[mask][0]

        return

    def run_fba(self, model_idx, iter):
        """General function to run parsimonious FBA (pFBA) on a model and store the results.
        This function runs pFBA on a given model, checks if the solution is above a minimum growth objective,
        and stores the resulting fluxes in the provided DataFrame.
        """
        # run pFBA
        sol1 = self.models[model_idx].slim_optimize()
        if sol1 > GROWTH_MIN_OBJ:
            if self.objective == "pfba":
                sol = cb.flux_analysis.parsimonious.pfba(self.models[model_idx])
            elif self.objective == "fba":
                sol = self.models[model_idx].optimize()
                
            self.org_fluxes.loc[(model_idx, iter, 0), list(sol.fluxes.index)] = self.rel_abund[model_idx] * sol.fluxes.values
        # do nothing otherwise - already initiated as zeros!

        return

    def update_env(self, update_rate, iter):
        """Function to update the environment fluxes based on the results of pFBA.
        This function calculates the new environment fluxes by taking into account the
        contributions of each organism's fluxes, weighted by their relative abundances.
        """
        #pull iter info and establish array shapes
        env_tmp = self.env_fluxes.loc[iter, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        #self.rel_abund  # (n_org, ) -> (n_org, 1)

        # get org fluxes
        total_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, 1) sum across orgs

        # check if environment fluxes are under-saturated
        over_shoot = np.zeros_like(total_org_flux)
        over_shoot[env_tmp != 0] = -total_org_flux[np.abs(env_tmp) >= 1e-6] / env_tmp[np.abs(env_tmp) >= 1e-6]

        # check if iteration uses more flux than available in environment
        if over_shoot.max().round(ROUND) <= 1:
            self.env_fluxes.loc[iter+1, 0] = (env_tmp + total_org_flux).flatten().round(ROUND) # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1) 
            is_under, update_rate = True, None
        else:
            ex_over = np.argmax(over_shoot) # index of flux causing over-shoot
            update_rate = (run_exs * env_tmp[ex_over, -1]) / (run_exs[ex_over, :] @ self.rel_abund)  #
            update_rate = update_rate.T
            is_under = False

        return is_under, update_rate, self.env_fluxes

    def iifba(self, rel_abund=None, iters=None, objective=None, early_stop=None, v=None):
        """Wrapper function for running iiFBA with parsimonious FBA (pFBA). This function initializes 
        the environment and organism fluxes, sets the exchange reactions for each model, runs pFBA,
        and updates the environment fluxes based on the results of pFBA. It returns the
        updated environment fluxes & organism fluxes.
        """
        # can change relative abundance, iters, or objective here
        if iters is not None:
            self.iters = utils.check_iters(iters)
            self.create_variables()
        if rel_abund is not None:
            self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        if objective is not None:
            self.objective = utils.check_objective(objective)
        if v is not None:
            self.v = v if not v else True
        if early_stop is not None:
            self.early_stop = early_stop if early_stop else False
        
        # iterate through iters with early stopping
        iter=0
        convergence = False # will stay false until convergence is achieved (no change to convergence if no early stop)
        while iter < self.iters and not(convergence):
            if self.v: print("Iteration:", iter)

            is_under = False
            low_bounds =  np.tile(self.env_fluxes.loc[iter, 0].to_numpy(), (self.size, 1))  # initialize update rate (used to scale ex flux bounds)
            for org_idx in range(self.size):
                low_bounds[org_idx, :] = low_bounds[org_idx, :] / self.rel_abund[org_idx]
            
            while not(is_under): # run until environment fluxes are under-saturated

                for org_idx in range(self.size):
                    if self.v: print(f"Running FBA for model {org_idx+1}/{self.size}...")
                    # set exchanges
                    self.set_env(org_idx, low_bounds[org_idx])

                    # run optim
                    self.run_fba(org_idx, iter)

                # update fluxes
                is_under, low_bounds, self.env_fluxes = self.update_env(low_bounds, iter)
                
                if self.v:
                    if not(is_under): print("Re-running Optimization due to over-saturation of environment fluxes.")

            # check for convergence for early stopping
            iter += 1
            if self.early_stop:
                last_iter = self.env_fluxes.loc[iter-1, 0].to_numpy().flatten()
                current_iter = self.env_fluxes.loc[iter, 0].to_numpy().flatten()
                if np.any(np.abs(last_iter - current_iter) > 1e-6):
                    convergence = False
                else:
                    convergence = True
                    self.env_fluxes.loc[(slice(iter+1, None),0), :] = self.env_fluxes.loc[(iter,0), :].values  # copy last iter to all future iters
                    iter = self.iters
                    if self.v: print("Convergence achieved.")
            

        # pfba has no use for Run index
        self.env_fluxes = self.env_fluxes.droplevel("Run")
        self.org_fluxes = self.org_fluxes.droplevel("Run")

        return self.env_fluxes, self.org_fluxes
    
    def summary(self, iter_shown=None):
        summary = CommunitySummary(self, iter_shown)

        return summary
    

