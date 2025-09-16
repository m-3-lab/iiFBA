import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from . import utils
from .config import GROWTH_MIN_OBJ, ROUND
from .summary import CommunitySummary

class iifbaObject:
    def __init__(self, models, media, rel_abund="equal", id=None):
        self.models = utils.check_models(models)
        self.media = media
        self.media = utils.check_media(self)
        self.size = len(self.models)
        self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        self.id = id

    def run_iifba(self, iters, method, early_stop=True, v=False):
        """_summary_
        Run the iifba algorithm

        """
        self.iters = utils.check_iters(iters)
        self.method = utils.check_method(method)
        self.early_stop = early_stop
        self.v = v

        # create variables
        self.create_vars()

        # run iterations
        for iter in range(self.iters):
            if self.v: print("Iteration:", iter)

            # update media for the iteration
            self.update_media(iter)

        # drop run col
        self.org_fluxes = self.org_fluxes.droplevel("Run")

        return self.env_fluxes, self.org_fluxes

    def create_vars(self, m_vals=[1,1]):
        """Initialize variables for iiFBA.
        This function sets up the optimization variables for the iiFBA analysis.
        """
        self.m_vals = m_vals # default to [1,1] for community iifba, can be set to [n, m] for sampling via iifba_sampling m_vals arg

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
        self.media = utils.check_media(self)
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

        # store model names
        self.model_names = {model_idx: model.name for model_idx, model in enumerate(self.models)}

        return
        
    def update_media(self, iter ):
        """
        Update the media (f_n,j) for each iteration
        """
        # define env bounds per organism for the current iteration
        env_bounds =  np.tile(self.env_fluxes.loc[iter, 0].to_numpy(), (self.size, 1))  # initialize update rate (used to scale ex flux bounds)
        for org_idx in range(self.size):
            env_bounds[org_idx, :] = env_bounds[org_idx, :] / self.rel_abund[org_idx]

        # run organism flux function
        self.flux_function(iter, env_bounds)

        # update media: f_n+1 = f_n - sum(v_nij)
        env_tmp = self.env_fluxes.loc[iter, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        sum_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, 1) sum across orgs

        self.env_fluxes.loc[iter+1, 0] = (env_tmp + sum_org_flux).flatten().round(ROUND) # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1)


        return

    def flux_function(self, iter, env_bounds):
        """
        run through flux function for organisms
        """
        # simulate each organism
        for model_idx in range(self.size):
            if self.v: print(" Simulating model:", model_idx+1, " of ", self.size)
            # set media
            self.set_env(model_idx, env_bounds[model_idx])

            # simulate each org
            self.sim_fba(model_idx, iter)

        # check over consumption
        self.check_overconsumption(iter, env_bounds)

        # once all orgs have been simulated without overconsumption, return to update_media
        return

    def set_env(self, model_idx, env_bounds):
        """
        Function to set the exchange reactions of a model to match the environment fluxes
        for a given iteration and run. This is mainly provided to ensure a cleaner wrapper function.
        """
        for ex in self.models[model_idx].exchanges:
            mask = np.array(self.org_exs) == ex.id
            if mask.any():  # Check if the exchange reaction exists in org_exs
                ex.lower_bound = -env_bounds[mask][0]

        return
    
    def sim_fba(self, model_idx, iter):
        """General function to run parsimonious FBA (pFBA) on a model and store the results.
        This function runs pFBA on a given model, checks if the solution is above a minimum growth objective,
        and stores the resulting fluxes in the provided DataFrame.
        """
        # run pFBA
        sol1 = self.models[model_idx].slim_optimize()
        if sol1 > GROWTH_MIN_OBJ:
            if self.method == "pfba":
                sol = cb.flux_analysis.parsimonious.pfba(self.models[model_idx])
            elif self.method == "fba":
                sol = self.models[model_idx].optimize()
                
            self.org_fluxes.loc[(model_idx, iter, 0), list(sol.fluxes.index)] = self.rel_abund[model_idx] * sol.fluxes.values
        # do nothing otherwise - already initiated as zeros!
        return
    
    def check_overconsumption(self, iter, adjusted_bounds):
        """
        Check over-consumption of env. mets. If over-consumption occurs, 
        re-run flux function (recursive subroutine)
        """
        #pull iter info and establish array shapes
        env_tmp = self.env_fluxes.loc[iter, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        #self.rel_abund  # (n_org, 1)

        # get org fluxes
        total_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, 1) sum across orgs

        # check if environment fluxes are under-saturated
        is_overconsumed = np.zeros_like(total_org_flux)
        is_overconsumed[env_tmp != 0] = -total_org_flux[np.abs(env_tmp) >= 1e-6] / env_tmp[np.abs(env_tmp) >= 1e-6]

        # check if iteration uses more flux than available in environment
        if is_overconsumed.max().round(ROUND) > 1:
            ex_over = np.argmax(is_overconsumed) # index of flux causing over-consumed
            # adjust only over-consumed bound
            # print(self.env_fluxes.columns[ex_over], "over-consumed by :", max(is_overconsumed))
            # print("fnij*:", env_tmp[ex_over, -1])
            # print("V'nij*:", run_exs[ex_over, :])
            # print("fnij* dot V'nij*:", env_tmp[ex_over, -1]* run_exs[ex_over, :])
            # print("V*nij*:", update_rate[:, ex_over])
            adjusted_bounds[:, ex_over] = (env_tmp[ex_over, -1] * run_exs[ex_over, :] / (run_exs[ex_over, :] @ self.rel_abund)).T #
            # print("V*nij*:", update_rate[:, ex_over])

            # re-run flux function with adjusted bounds
            if self.v: print("Re-running Optimization due to over-saturation of environment fluxes.")
            self.flux_function(iter, adjusted_bounds)

        return
