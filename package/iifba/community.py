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

    def __init__(self, models, media, rel_abund=None, iters=10, objective= "pfba", early_stop=True, v=False):
        # input parameters validation
        self.models = utils.check_models(models)
        self.size = len(models)
        self.media = media
        self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        self.iters = utils.check_iters(iters)
        self.objective = utils.check_objective(objective)
        self.id = None # for plotting
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

    def create_variables(self, m_vals=[1,1]):
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

        return is_under, update_rate

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
                is_under, low_bounds = self.update_env(low_bounds, iter)

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
                    if iter <= self.iters - 1:# copy last iter to all future iters if, converging not on last
                        self.env_fluxes.loc[(slice(iter+1, None),0), :] = self.env_fluxes.loc[(iter,0), :].values  # copy last iter to all future iters
                    iter = self.iters
                    if self.v: print("Convergence achieved.")
            

        # pfba has no use for Run index
        self.env_fluxes = self.env_fluxes.droplevel("Run")
        self.org_fluxes = self.org_fluxes.droplevel("Run")

        return self.env_fluxes, self.org_fluxes
    
    def summarize(self, iter_shown=None):
        self.summary = CommunitySummary(self, iter_shown)

        return self.summary

    def run_sampling(self, model_idx, iter, rep_idx, Mi, runs_over):
        # add minimum growth constraint to model
        min_obj = self.models[model_idx].slim_optimize() * self.objective_percent
        sol = self.models[model_idx].optimize()
        obj_rxn = [rxn.id for rxn in linear_reaction_coefficients(self.models[model_idx]).keys()][0]
        self.models[model_idx].reactions.get_by_id(obj_rxn).lower_bound = min_obj

        #run flux sampling
        samples = sum(runs_over)
        if min_obj >= GROWTH_MIN_OBJ:
            sol = cb.sampling.sample(self.models[model_idx], samples)

            # store fluxes
            arrays = [[model_idx] * samples, [iter] * samples, list(np.where(runs_over == 1)[0])]
            tuples = list(zip(*arrays))
            multi_idx = pd.MultiIndex.from_tuples(tuples, names=['Model', 'Iteration', 'Run'])
            sol.index = multi_idx
            
            self.org_fluxes.loc[sol.index, sol.columns] = self.rel_abund[model_idx] * sol.values
            if iter > 0:
                self.cumulative_fluxes.loc[sol.index, sol.columns] = self.org_fluxes.loc[sol.index, sol.columns] + self.cumulative_fluxes.loc[(model_idx, iter-1, Mi)][sol.columns]
            else:
                self.cumulative_fluxes.loc[sol.index, sol.columns] = self.org_fluxes.loc[sol.index, sol.columns]

    def update_sampling_env(self, iter, Mi, rep_idx, init_sample_ct, runs_over):
        """Function to update the environment fluxes based on the results of flux sampling. 
        This is similar to the update_env function but adapted for flux sampling.
        """
        # repeat update per sample
        for sample_idx in np.where(runs_over == 1)[0]: # only update for runs that are over
            # get env fluxes
            env_tmp = self.env_fluxes.loc[iter, Mi][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
            run_exs = self.org_fluxes.loc[:, iter, rep_idx * init_sample_ct + sample_idx][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux

            # get org fluxes
            total_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, 1) sum across orgs

            # check if environment fluxes are under-saturated
            over_shoot = np.zeros_like(total_org_flux)
            over_shoot[env_tmp != 0] = -total_org_flux[np.abs(env_tmp) >= 1e-6] / env_tmp[np.abs(env_tmp) >= 1e-6]

            # check if iteration uses more flux than available in environment
            if over_shoot.max().round(ROUND) <= 1:
                self.env_fluxes.loc[iter+1, rep_idx * init_sample_ct  + sample_idx] = (env_tmp + total_org_flux).flatten().round(ROUND) # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1)
                runs_over[sample_idx] = 0

        return runs_over

    def iifba_sampling(self, m_vals, iters=None, objective_percent=0.9, rel_abund=None, v=None, convergence=None):
        """
        v input validation
        objective_percent input validation
        """
        # can change relative abundance, iters, or objective here
        if iters is not None:
            self.iters = utils.check_iters(iters)
        if rel_abund is not None:
            self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        if v is not None:
            self.v = v if not v else True
        self.sampling_convergence = convergence
        self.m_vals, self.sampling_convergence, self.iters = utils.check_sampling_inputs(self, m_vals)
        self.objective_percent = objective_percent

        # re-create variables for sampling
        self.create_variables(m_vals)
        self.cumulative_fluxes = self.org_fluxes.copy()  # initialize cumulative fluxes

        # mapping of what flux sampling to iterate
        self.M = np.zeros([self.m_vals[0], self.iters],dtype=int) #randomly pre-assign sampling initial point matrix
        for i in range(1, self.iters):
            Mcol = np.sort(np.random.choice(self.m_vals[0]*self.m_vals[1],self.m_vals[0],replace=False))
            self.M[:,i]= Mcol
        
        iter = 0
        is_converged = False # will stay false until convergence is achieved (no change to convergence if no early stop)
        while iter < self.iters and not(is_converged):
            if self.v: print("Iteration:", iter)

            # number of times to re-sample per iteration
            start_num = 1 if iter == 0 else self.m_vals[0]
            for rep_idx in range(start_num):


                # check if starting point is valid
                max_Mi_retries = 10
                max_Mi_retries = np.clip(max_Mi_retries, 0, self.m_vals[0])  # ensure max retries is within bounds
                exclude = set(self.M[:, iter])
                if iter != 0:
                    for _ in range(max_Mi_retries):
                        try:
                            Mi = self.M[rep_idx, iter]
                            while Mi in exclude:
                                all_possible_Mis = set(range(self.m_vals[0]*self.m_vals[1]))
                                available_Mis = list(all_possible_Mis - exclude)       
                                if not available_Mis:
                                    is_converged = True
                                Mi = np.random.choice(available_Mis)
                            exclude.add(Mi)
                            self.M[rep_idx, iter] = Mi  # update M with valid Mi

                            low_bounds =  np.tile(self.env_fluxes.loc[iter, Mi].to_numpy(), (self.size, 1))
                            for org_idx in range(self.size):
                                low_bounds[org_idx, :] = low_bounds[org_idx, :] / self.rel_abund[org_idx]
                            for org_idx in range(self.size):
                                # set exchanges
                                self.set_env(org_idx, low_bounds[org_idx])
                                # run optimization 
                                self.run_sampling(org_idx, iter, rep_idx, Mi, runs_over)
                            break  # exit retry loop if successful (Mi is a valid start point)
                        except:
                            continue
                    else:
                        is_converged = True  # if all retries failed, set convergence to True to exit loop
                else:
                    Mi = self.M[rep_idx, iter]

                # set up low_bounds
                resample_ct = 0
                init_sample_ct = self.m_vals[0] * self.m_vals[1] if iter == 0 else self.m_vals[1]
                runs_over = np.ones(init_sample_ct, dtype=int)  # initialize runs_over to track which runs are over-saturated
                low_bounds =  np.tile(self.env_fluxes.loc[iter, Mi].to_numpy(), (self.size, 1))
                for org_idx in range(self.size):
                    low_bounds[org_idx, :] = low_bounds[org_idx, :] / self.rel_abund[org_idx]

                # re-sample until env is under-saturated or max resample reached. reaching max re-sample will just move to next set of runs keeping env at 0 and subsequent growth at 0
                while sum(runs_over) != 0 and resample_ct < 50:
                    for org_idx in range(self.size):
                        # set exchanges
                        self.set_env(org_idx, low_bounds[org_idx])
                        # run optimization 
                        self.run_sampling(org_idx, iter, rep_idx, Mi, runs_over)

                    # update fluxes
                    runs_over = self.update_sampling_env(iter, Mi, rep_idx, init_sample_ct, runs_over)
                    resample_ct += 1
                    if self.v:
                        if sum(runs_over) > 0: print(f"Re-running Optimization due to over-saturation of environment fluxes. ({sum(runs_over)} fluxes are over)")

            # check for convergence for early stopping
            if self.sampling_convergence is not None:
                # check distribution of fluxes
                if self.v: print("Checking for convergence...")
                is_converged = self.check_sampling_convergence(iter)

            iter += 1 
        return
    
    def check_sampling_convergence(self, iter):


        is_converged = False
        # run logistic regression to check for convergence 
        if "logistic" in self.sampling_convergence and iter > 0:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            # use flux values as features and iteration as target (more difference == higher accuracy)
            iterations_to_use = [iter, iter+1]
            idx = pd.IndexSlice
            X = self.env_fluxes.loc[idx[iterations_to_use, :], :]
            y = X.index.get_level_values('Iteration')
            y = y.map({iterations_to_use[0]: 0, iterations_to_use[1]: 1})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # run log regression, higher accuracy = more  differentiation between iterations
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            if accuracy <= self.sampling_convergence["logistic"]:
                if self.v: print("Converged at Iteration", iter, "with Accuracy:", accuracy, "(Lower accuracy = Higher Similarity)")
                is_converged = True

        # use energy distance -- issues with SciPy :(
        # elif "energy" in self.sampling_convergence and iter > 0:
        #     from scipy.stats import energy_distance
        #     # get the 2 iterations to claculate energy distance
        #     df_iter0 = self.env_fluxes.xs(iter, level='Iteration').to_numpy()
        #     df_iter1 = self.env_fluxes.xs(iter+1, level='Iteration').to_numpy()

        #     # calculate energy distance
        #     energy_dist = energy_distance(df_iter0, df_iter1)
        #     if iter == 1: 
        #         self.energy_dist_prev = energy_dist
        #     if iter > 1:
        #         if (energy_dist - self.energy_dist_prev)/self.energy_dist_prev < self.sampling_convergence["energy"]:
        #             is_converged = True
        #     if self.v: print("Converged at Iteration", iter, "with Accuracy:", accuracy)
        #     self.energy_dist_prev = energy_dist

        elif "ks" in self.sampling_convergence and iter > 0:
            from scipy.stats import ks_2samp
            from statsmodels.stats.multitest import multipletests
            # get the 2 iterations to claculate energy distance
            df_iter0 = self.env_fluxes.xs(iter, level='Iteration')
            df_iter1 = self.env_fluxes.xs(iter+1, level='Iteration')

            # run ks_2samp for each feature/reaction
            p_vals = []
            for feature in df_iter0.columns:
                _, p_value = ks_2samp(df_iter0[feature], df_iter1[feature])
                p_vals.append(p_value)

            # higher rejects = more similar = convergence
            reject, _, _, _ = multipletests(p_vals, method='fdr_bh')
            if 1-reject.mean() > self.sampling_convergence["ks"]:
                if self.v: print("Converged at Iteration", iter, "with KS-Based Score:", 1-reject.mean(), "(Higher KS = Higher Similarity)")
                is_converged = True

        return is_converged