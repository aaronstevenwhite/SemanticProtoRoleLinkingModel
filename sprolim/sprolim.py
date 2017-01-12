import sys
import cPickle as pickle
import numpy as np
import scipy as sp
import pandas as pd
import theano
import theano.tensor as T

from itertools import combinations, combinations_with_replacement, permutations
from scipy.misc import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.cross_validation import LabelKFold, train_test_split

from utility import *
from data import Data

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action = "ignore")

#theano.config.profile = True

class Sprolim(object):
    
    def __init__(self, data, k=2, alpha=1., ycats=['subject', 'object', 'oblique'],
                 initialized_model=None, semantics_only=False):
        self.data = data
        self.k = k if initialized_model is None else initialized_model.k
        self.alpha = alpha if initialized_model is None else initialized_model.alpha
        
        self.ycats = np.array(ycats)
        self.initialized_model = initialized_model
        self.semantics_only = semantics_only
        
        self.ident = ''.join([str(i) for i in np.random.choice(9, size=10)])

        self._initialize_data()
        self._initialize_model()
        
    def _initialize_data(self):
        
        self._create_syntactic_predictors()

        maxargs = self.data.groupby('roleset').argnum.apply(np.max).reset_index()
        self.data = pd.merge(self.data, maxargs.rename(columns={'argnum' : 'maxargnum'}))
        self.max_ltrs = np.max(self.data.maxargnum)

        self.data['argposrel'] = self.data.groupby(['sentenceid', 'roleset']).argposud.apply(get_codes)
        
        self.data_split = {i : Data(self.data[self.data.maxargnum==i], self.k, self.ycats) for i in self.data.maxargnum.unique()}

        self.num_of_properties = get_num_of_types(self.data.property)
        self.num_of_syntpos = get_num_of_types(self.data.gramfuncud)
                

    def _create_syntactic_predictors(self):
        self.data['indicator'] = 1

        ind = ['sentenceid', 'roleset', 'predtokenud', 'argposud', 'property']
        col = ['gramfuncud']
        
        y_frame = self.data.pivot_table(index=ind,
                              columns=col,
                              values='indicator',
                              fill_value=0).reset_index()

        count_by_sent = self.data.pivot_table(index=ind[:2],
                                              columns=col,
                                              values='indicator',
                                              aggfunc=max,
                                              fill_value=0).reset_index()

        count_by_arg = self.data.pivot_table(index=ind,
                                             columns=col,
                                             values='indicator',
                                             aggfunc=sum,
                                             fill_value=0).reset_index()
        
        y_cast = pd.merge(count_by_sent, count_by_arg, on=ind[:2])
    
        self.data = pd.merge(self.data, y_cast, on=ind, how='outer')
        
        for position in self.ycats:
            self.data[position] = np.maximum(np.array(self.data[position+'_x'] - self.data[position+'_y']), 0)

        self.other_positions = self.data[self.ycats]

            
    def _initialize_mixture(self):
        role_ll = {}
        
        for i, data in self.data_split.iteritems():
            mixture_aux = np.zeros([data.num_of_rolesets_total, data.max_ltrs, self.k])

            self.representations['mixture_'+str(i)] = theano.shared(mixture_aux,
                                                                   name='mixture'+str(i)+self.ident)

            mixture_exp = T.exp(self.representations['mixture_'+str(i)])
            mixture = mixture_exp / T.sum(mixture_exp, axis=2)[:,:,None]

            log_mixture = T.log(T.swapaxes(mixture, 0, 2))

            role_perms_logprob = T.sum(log_mixture[data.role_perms,T.arange(i)[None,:]], axis=1)
            
            role_ll[i] = {}
            
            for j in data.unique_argnums:
                log_divisor = T.log(data.num_of_sents_per_roleset[data.roleset[j]])
                role_ll[i][j] = T.transpose(role_perms_logprob)[data.roleset[j]] -\
                                log_divisor[:,None]
                
        return role_ll, (self.alpha - 1.)*T.sum(log_mixture)

    def _initialize_canonicalization(self):
        canon_ll = {}
        
        for i, data in self.data_split.iteritems():
            canon_ll[i] = {}
            
            for j in data.unique_argnums:
                # canonicalization_prob_aux = np.random.normal(0., 1., [data.num_of_rolesets[j],
                #                                                       data.num_of_ltr_perms[j]])

                canonicalization_prob_aux = np.tile(1./np.arange(1, data.num_of_ltr_perms[j]+1),
                                                    [data.num_of_rolesets[j], 1])
                
                self.representations['canonicalization_'+str(i)+str(j)] = theano.shared(canonicalization_prob_aux,
                                                                                name='canonicalization_prob'+str(i)+str(j)+self.ident)

                canonicalization_exp = T.exp(self.representations['canonicalization_'+str(i)+str(j)])
                canonicalization_prob = canonicalization_exp / T.sum(canonicalization_exp, axis=1)[:,None]
                canonicalization_logprob = T.log(canonicalization_prob)[data.rolesettokenindices[j]]

                canon_ll[i][j] = canonicalization_logprob[data.sentroleset[j],None,:] - T.log(j)
            
        return canon_ll

    def _initialize_applicability(self):
        if self.initialized_model is None:
            #appl_aux = np.random.normal(0, 1, size=[self.k, self.num_of_properties, 2])
            appl_aux = np.zeros([self.k, self.num_of_properties, 2])
        else:
            appl_aux = self.initialized_model.representations['applicable'].eval()

        self.representations['applicable'] = theano.shared(appl_aux, name='appl'+self.ident)

        appl_exp = T.exp(self.representations['applicable'])
        appl_prob = appl_exp / T.sum(appl_exp, axis=2)[:,:,None]

        appl_ll = {}
        
        for i, data in self.data_split.iteritems():
            appl_ll[i] = {}
            for j in data.unique_argnums:
                appl_ll[i][j] = T.log(appl_prob[data.position_role_map[j][data.argposrel[j]],
                                             data.property[j][:,None,None],
                                             data.applicable[j][:,None,None]])

        return appl_ll

    def _initialize_rating(self):
        if self.initialized_model is None:
            #rating_aux = np.random.normal(0, 1, size=[self.k, self.num_of_properties])
            rating_aux = np.zeros([self.k, self.num_of_properties])
        else:
            rating_aux = self.initialized_model.representations['rating'].eval()

        self.representations['rating'] = theano.shared(rating_aux, name='rating'+self.ident)

        if self.initialized_model is None:
            jumps_aux = np.array([-np.inf] + [1]*(np.max(self.data.response)-1) + [-np.inf])
        else:
            jumps_aux = self.initialized_model.representations['jumps'].eval()
            
        self.representations['jumps'] = theano.shared(jumps_aux, name='jumps'+self.ident)

        jumps = T.exp(self.representations['jumps'])
        cuts = T.extra_ops.cumsum(jumps)
        cuts_centered = cuts - cuts[(np.max(self.data.response)-1)/2]

        rating_logprob_cum = cuts_centered[None,None,:] - self.representations['rating'][:,:,None]


        rating_prob_cum = (1. / (1. + T.exp(-rating_logprob_cum)))

        rate_ll = {}
        
        for i, data in self.data_split.iteritems():
            rate_ll[i] = {}
            for j in data.unique_argnums:
                rate_prob_cum_high = rating_prob_cum[data.position_role_map[j][data.argposrel[j]],
                                                     data.property[j][:,None,None],
                                                     data.response[j][:,None,None]]
                rate_prob_cum_low = rating_prob_cum[data.position_role_map[j][data.argposrel[j]],
                                                    data.property[j][:,None,None],
                                                    data.response[j][:,None,None]-1]

                rate_prob_high = T.switch(T.lt(T.tile(data.response[j][:,None,None],
                                                      [1,data.role_perms.shape[0],
                                                       data.num_of_ltr_perms[j]]),
                                               self.data.response.max()),
                                          rate_prob_cum_high,
                                          T.ones_like(rate_prob_cum_high))

                rate_prob_low = T.switch(T.gt(T.tile(data.response[j][:,None,None],
                                                     [1,data.role_perms.shape[0],
                                                      data.num_of_ltr_perms[j]]),
                                              self.data.response.min()),
                                         rate_prob_cum_low,
                                         T.zeros_like(rate_prob_cum_high))

                rate_prob = rate_prob_high - rate_prob_low

                rate_ll[i][j] = data.applicable[j][:,None,None]*T.log(rate_prob)

        return rate_ll
    
    def _initialize_syntax(self):
        if self.initialized_model is None:
            role_synt_aux = np.random.normal(0., 1., size=[self.num_of_syntpos,self.k])
            synt_synt_aux = LogisticRegression(fit_intercept=False,
                                   multi_class='multinomial',
                                   solver='newton-cg').fit(self.other_positions,
                                                           self.data.gramfuncud).coef_

        else:
            role_synt_aux = self.initialized_model.representations['role_syntax'].eval()
            synt_synt_aux = self.initialized_model.representations['syntax_syntax'].eval()
            
        self.representations['role_syntax'] = theano.shared(role_synt_aux, name='role_synt'+self.ident)
        role_synt = self.representations['role_syntax']

        self.representations['syntax_syntax'] = theano.shared(synt_synt_aux, name='synt_synt'+self.ident)
        synt_synt = self.representations['syntax_syntax']

        synt_ll = {}
        
        for i, data in self.data_split.iteritems():
            synt_ll[i] = {}
            
            for j in data.unique_argnums:                
                synt_synt_sum = T.sum(data.other_positions[j][:,None,:]*synt_synt[None,:,:], axis=2)
                synt_potential = synt_synt_sum[:,:,None]+role_synt[None,:,:]

                synt_potential_exp = T.exp(synt_potential)
                synt_prob = synt_potential_exp / T.sum(synt_potential_exp, axis=1)[:,None,:]
                

                synt_ll[i][j] = synt_prob[T.arange(data.data[j].shape[0])[:,None,None],
                                          data.gramfuncud[j][:,None,None],
                                          data.position_role_map[j][data.argposrel[j]]]
                synt_ll[i][j] = synt_ll[i][j] - T.log(float(self.num_of_properties))
 
        return synt_ll
            
    
    def _initialize_model(self):

        self.representations = {}
        
        ## mixture
        role_ll, role_prior_ll = self._initialize_mixture()
        
        ## canonicalization
        canonicalization_ll = self._initialize_canonicalization()
        
        ## applicability components
        appl_ll = self._initialize_applicability()
        
        ## rating components
        rating_ll = self._initialize_rating()
        
        ## syntactic position components
        syntax_ll = self._initialize_syntax()

        self.total_loglike_sum = 0.
        self.total_loglike_sum_synt_only = 0.

        self._role_perm = {}
        
        for i, data in self.data_split.iteritems():
            self._role_perm[i] = {}
            
            for j in data.unique_argnums:
                ## likelihood
                inner_sum = canonicalization_ll[i][j]+appl_ll[i][j]+rating_ll[i][j]
                
                if not self.semantics_only:
                    inner_sum += syntax_ll[i][j]

                self._role_perm[i][j] = role_ll[i][j][:,:,None] + inner_sum
                    
                perm_ll = T.log(T.sum(T.exp(inner_sum), axis=2))
                
                self.total_loglike_sum += T.sum(T.log(T.sum(T.exp(role_ll[i][j] + perm_ll), axis=1)))

                inner_sum_synt_only = canonicalization_ll[i][j]+syntax_ll[i][j]
                perm_ll_synt_only = T.log(T.sum(T.exp(inner_sum_synt_only), axis=2))

                self.total_loglike_sum_synt_only += T.sum(T.log(T.sum(T.exp(role_ll[i][j] + perm_ll_synt_only), axis=1)))
                
        self.total_loglike_sum += role_prior_ll
                
    def _initialize_updaters(self, fix_params):
        update_dict_gd = []
        update_dict_ada = []

        self.rep_grad_hist_t = {}
        
        for name, rep in self.representations.items():            
            if not fix_params or name.split('_')[0] in ['canonicalization', 'mixture']:
                rep_grad = T.grad(self.total_loglike_sum, rep)

                # if self.initialized_model is None or name.split('_')[0] in ['canonicalization', 'mixture']:
                self.rep_grad_hist_t[name] = theano.shared(np.ones(rep.shape.eval()), name=name+'_hist'+self.ident)
                # else:
                #     self.rep_grad_hist_t[name] = theano.shared(self.initialized_model.rep_grad_hist_t[name].eval(),
                #                                                name=name+'_hist'+self.ident)

                rep_grad_adj = rep_grad / (T.sqrt(self.rep_grad_hist_t[name]))
                
                if name == 'jumps':
                    learning_rate = 0.000001
                else:
                    learning_rate = 0.00001

                update_dict_gd += [(self.rep_grad_hist_t[name], self.rep_grad_hist_t[name] + T.power(rep_grad, 2)),
                                    (rep, rep + learning_rate*rep_grad)]
                update_dict_ada += [(self.rep_grad_hist_t[name], self.rep_grad_hist_t[name] + T.power(rep_grad, 2)),
                                    (rep, rep + rep_grad_adj)]  
            
        self.updater_gd = theano.function(inputs=[],
                                          outputs=[self.total_loglike_sum,
                                                   self.total_loglike_sum_synt_only],
                                          updates=update_dict_gd,
                                          name='updater_gd_'+self.ident)

        self.updater_ada = theano.function(inputs=[],
                                           outputs=[self.total_loglike_sum,
                                                    self.total_loglike_sum_synt_only],
                                           updates=update_dict_ada,
                                           name='updater_ada'+self.ident)


    def fit(self, iterations=1000, tolerance=1., gd_init=True, fix_params=False, verbose=False):

        self._initialize_updaters(fix_params)
        
        prev_ll = -np.inf
        
        for i in range(iterations):
            if i < 5 and gd_init:
                self.ll, self.ll_synt_only = self.updater_gd()
            else:
                self.ll, self.ll_synt_only = self.updater_ada()
        
            if verbose:
                print '{:<3d}\t{:08.2f}\t{:08.2f}'.format(int(i),
                                                           float(self.ll),
                                                           float(self.ll_synt_only))

            if (self.ll - prev_ll) >= tolerance:
                prev_ll = self.ll
            else:
                break

        print
            
        #self.predict()
                
        return self
                
    def compute_deviance_aic(self, syntax_only=False):
        if syntax_only:
            num_params = [np.prod(rep.shape.eval()) for name, rep in self.representations.items()
                          if name not in ['applicable', 'rating', 'jumps']]
            
            return -2*self.ll_synt_only, -2*self.ll_synt_only + 2*np.sum(num_params)

        else:
            num_params = [np.prod(rep.shape.eval()) for rep in self.representations.values()]

            return -2*self.ll, -2*self.ll + 2*np.sum(num_params)
    
    @property
    def mixture(self):
        mixture_dict = {}
        
        for i, data in self.data_split.iteritems():
            mixture_exp = T.exp(self.representations['mixture_'+str(i)])
            mixture = (mixture_exp / T.sum(mixture_exp, axis=2)[:,:,None]).eval()

            mixture_dict[i] = pd.Panel(mixture,
                                          items=data.rolesets_list,
                                          major_axis=['ltr'+str(j) for j in range(data.max_ltrs)],
                                          minor_axis=['role'+str(j) for j in range(self.k)])

        return mixture_dict
            
    @property
    def canonicalization(self):
        canon_dict = {}
        
        for i, data in self.data_split.iteritems():
            canon_dict[i] = {}
            for j in data.unique_argnums:
                canon_exp = T.exp(self.representations['canonicalization_'+str(i)+str(j)])
                canon_prob = (canon_exp / T.sum(canon_exp, axis=1)[:,None])

                canon_dict[i][j] = pd.DataFrame(canon_prob.eval(),
                                                index=data.data[j].roleset.astype('category').cat.categories)

        return canon_dict

    @property
    def mixture_canonicalized(self):
        mixtures = self.mixture
        canons = self.canonicalization

        mixcanon_list = []
        
        for i, data in self.data_split.iteritems():
            for j in data.unique_argnums:
                rsets = np.array(canons[i][j].index)

                mixtures_j = np.array(mixtures[i][rsets])

                max_canon = np.argmax(np.array(canons[i][j]), axis=1)
                ltr_perm_selected = data.ltr_perms[j][max_canon]
                
                mixcanon = pd.Panel(mixtures_j[np.arange(rsets.shape[0])[:,None],
                                                          ltr_perm_selected],
                                               items=rsets,
                                               major_axis=[str(k) for k in range(j)],
                                               minor_axis=['role'+str(k) for k in range(self.k)])
                                               

                mixcanon = pd.melt(mixcanon.to_frame().reset_index(),
                                   id_vars=['major', 'minor'],
                                   var_name='roleset',
                                   value_name='probability')
                mixcanon['argnum'] = j

                mixcanon = mixcanon.rename(columns={'major' : 'argposrel',
                                                    'minor' : 'role'})

                mixcanon = mixcanon.pivot_table(values='probability',
                                                index=['roleset', 'argnum', 'argposrel'],
                                                columns='role').reset_index()
                
                mixcanon_list.append(mixcanon)
                
        mixcanon_all = pd.concat(mixcanon_list)
        mixcanon_all.argposrel = mixcanon_all.argposrel.astype(int)

        return mixcanon_all
    
    @property
    def rating(self):
        return pd.DataFrame(np.transpose(self.representations['rating'].eval()),
                            index=self.data.property.astype('category').cat.categories)

    @property
    def applicable(self):
        appl_exp = T.exp(self.representations['applicable'])
        appl = (appl_exp / T.sum(appl_exp, axis=2)[:,:,None])

        return pd.DataFrame(np.transpose(appl[:,:,1].eval()),
                            index=self.data.property.astype('category').cat.categories)

    @property
    def role_syntax(self):
        return pd.DataFrame(self.representations['role_syntax'].eval(),
                            columns=['role'+str(i) for i in range(self.k)], index=self.ycats)

    @property
    def syntax_syntax(self):
        return pd.DataFrame(self.representations['syntax_syntax'].eval(), index=self.ycats, columns=self.ycats)

    def predict_ltr(self, maximize=False):

        prediction = []
        
        for i, data in self.data_split.iteritems():
            
            for j in data.unique_argnums:
                x = self._role_perm[i][j].eval()

                max_idx = x.reshape(x.shape[0],-1).argmax(1)
                maxpos_vect = np.column_stack(np.unravel_index(max_idx, x[0,:,:].shape))

                data_j = data.data[j]                
                data_j['arg_predicted'] = data.ltr_perms[j][maxpos_vect[:,1], data.data[j].argposrel]

                prediction.append(data_j)

        return pd.concat(prediction)
    
    def predict(self, maximize=False):
        ## extract the role-syntax and syntax-syntax parameters
        params_Xy = self.representations['role_syntax'].eval()
        params_yy = self.representations['syntax_syntax'].eval()

        ## extract the mixture and canonicalization probabilities
        mixtures = self.mixture
        canons = self.canonicalization
        
        ll_sum = {}

        result = []
        prediction = {}
        
        for i, data in self.data_split.iteritems():
            prediction[i] = {}
            
            for j in data.unique_argnums:
                
                ## construct all i-combinations of the syntactic position indices
                sent_combs = list(combinations_with_replacement(range(self.ycats.shape[0]), j))

                ## construct all permutations of each i-combination of the syntactic position indices
                sent_perms = np.array(list({perm for comb in sent_combs for perm in permutations(comb)}))

                ## count how many times each syntactic position occurs in each permutation
                ## creates a permutation-by-syntactic position matrix
                bin_counter = lambda x: np.bincount(x, minlength=self.ycats.shape[0])
                sent_perm_counts = np.array(map(bin_counter, sent_perms))

                ## for each argument position in each permutation, subtract the count of 
                ## the syntactic position that labels that argument position
                ## creates a permutation-by-argument position-by-syntactic position tensor A
                ## where A[i,j] gives the vector of syntactic position counts after subtracting
                ## the count for the syntactic position labeling argument j in permutation i 
                sent_perm_counts = sent_perm_counts[:,None,:] -\
                                   np.eye(self.ycats.shape[0])[sent_perms]

                ## create a permutation-by-argument position-by syntactic position tensor A
                ## where A[i,j] is a vector of potentials for each argument position j in
                ## permutation i representing the strength of belief we have that argument j
                ## has a particular syntactic position
                synt_synt_pot = np.sum(params_yy[None,None,:,:] *\
                                       sent_perm_counts[:,:,None,:],
                                       axis=3)
            
                ## create a permutation-by-argument position-by-syntactic position-by role tensor
                ## A where A[i,j,:,k] is a vector of potentials for each argument position j in
                ## permutation i representing the strength of belief we have that argument j
                ## has a particular syntactic position given the role of argument j is k
                potentials = params_Xy[None,None,:,:] + synt_synt_pot[:,:,:,None]

                ## create a permutation-by-argument position-by-role matrix A by getting to "true"
                ## syntactic position for each argument position in each permutation; A[i,j,k] gives
                ## the probability that argument j in permutation i has the syntactic position
                ## listed in that permutation given that it has role k
                synt_logprobs = np.log(np.exp(potentials) / np.sum(np.exp(potentials),
                                                                   axis=2)[:,:,None,:])
                synt_logprobs = synt_logprobs[np.arange(sent_perms.shape[0])[:,None],
                                              np.arange(sent_perms.shape[1])[None,:],
                                              sent_perms]

                rsets = np.array(canons[i][j].index)

                mixtures_j = np.array(np.log(mixtures[i][rsets]))

                if maximize:
                    max_canon = np.argmax(np.array(canons[i][j]), axis=1)
                    ltr_perm_selected = data.ltr_perms[j][max_canon]

                    mixture_selected = mixtures_j[np.arange(rsets.shape[0])[:,None],
                                                  ltr_perm_selected]

                    # comment this
                    # mixture_maxed = np.transpose(np.eye(j)[np.argmax(mixture_selected, axis=1)], (0,2,1))
                    # mixture_maxed += (1-np.sum(mixture_maxed, axis=2))[:,:,None]*mixture_selected

                    mixture_maxed = np.eye(self.k)[np.argmax(mixture_selected, axis=2)]
                    
                    sent_perm_prob = np.sum(synt_logprobs[None,:,:,:] * mixture_maxed[:,None,:,:],
                                            axis=(2,3))
                    # comment this
                    
                    # sent_perm_prob = np.sum(synt_logprobs[None,:,:,:] + mixture_selected[:,None,:,:],
                    #                         axis=(2,3))
                    
                    pred_ind = np.argmax(sent_perm_prob, axis=1)
                else:
                    pr_map = data.position_role_map[j].eval()

                    ## take the product (sum in log space) over the argument positions after mapping to label
                    ## creates a canonicalization-by-role labeling-by-permutation tensor A, where A[i,j,k]
                    ## give the probability of the syntactic position labeling k of arguments positions for
                    ## canonicalization i with role labeling j
                    sent_logprobs = np.sum(synt_logprobs.swapaxes(0,2)[pr_map,
                                                                       np.arange(pr_map.shape[0])[:,None,None]],
                                           axis=0)

                    mixture_logprob = mixtures_j[np.arange(rsets.shape[0])[:,None,None,None],
                                                 data.ltr_perms[j].T[None,:,None,:],
                                                 pr_map[None,:,:,:]]
                    canon_logprob = np.array(np.log(canons[i][j]))

                    role_canon_logprob = np.sum(mixture_logprob+canon_logprob[:,None,None,:], axis=1)

                    total_logprob = role_canon_logprob[:,:,:,None] + sent_logprobs[None,:,:,:]

                    sum_ind = logsumexp(total_logprob, axis=(1,2))
                    pred_ind = np.argmax(sum_ind, axis=1)
                    
                prediction[i][j] = pd.DataFrame(sent_perms[pred_ind],
                                                index=canons[i][j].index)

            ids = ['sentenceid', 'roleset', 'predtokenud', 'argposrel', 'argnum']
            argnums = data.data_all.groupby(ids).property.apply(len).reset_index().drop('property', axis=1)
        
        for i, dct in prediction.iteritems():
            for j, df in dct.iteritems():
                df_reset = df.reset_index().rename(columns={'index':'roleset'})
                df_melted = pd.melt(df_reset, id_vars='roleset',
                                    var_name='argposrel',
                                    value_name='gramfuncud_predicted')
                df_melted['argnum'] = j

                result.append(df_melted)

        result = pd.concat(result)
        
        result_merged = pd.merge(self.data, result)[ids+['gramfuncud', 'gramfuncud_predicted']]
        result_merged['gramfuncud'] = result_merged['gramfuncud'].astype('category',
                                                                         categories=self.ycats).cat.codes

        self._prediction = result_merged
        
        return self._prediction

    @property
    def prediction(self):
        return self._prediction

    @property
    def confusion(self):
        conf = confusion_matrix(self._prediction.gramfuncud,
                                     self._prediction.gramfuncud_predicted)/float(self.num_of_properties)

        return pd.DataFrame(conf,index=self.ycats,columns=self.ycats)
        

    @property
    def metrics(self):
        mets = precision_recall_fscore_support(self._prediction.gramfuncud,
                                               self._prediction.gramfuncud_predicted)

        return pd.DataFrame(np.array(mets),
                            index=['precision', 'recall', 'fscore', 'support'],
                            columns=self.ycats)

    def write(self):
        name = lambda x: x+'_retrained' if self.semantics_only else x

        self.rating.to_csv(name('rating')+str(self.k)+'.csv')
        self.applicable.to_csv(name('applicable')+str(self.k)+'.csv')
        self.syntax_syntax.to_csv(name('syntax_syntax')+str(self.k)+'.csv')
        self.role_syntax.to_csv(name('role_syntax')+str(self.k)+'.csv')
        self.mixture_canonicalized.to_csv(name('mixture_canonicalized')+str(self.k)+'.csv')
        self.metrics.to_csv(name('metrics')+str(self.k)+'.csv')
        self.prediction.to_csv(name('prediction')+str(self.k)+'.csv')
        self.confusion.to_csv(name('confusion')+str(self.k)+'.csv') 
    
class SprolimModelSelection(object):

    def __init__(self, data, rolenums=range(2,5)):
        self.data = data
        self.rolenums = rolenums

    def run_train_test(self, train_prop=0.9, init_tolerance=1000., train_tolerance=1., test_tolerance=0.1):
        if init_tolerance is not None:
            self._train_initialization(init_tolerance)
        
        train_roleset, test_roleset = train_test_split(self.data.roleset.unique(),
                                                     train_size=train_prop)

        data_train = self.data[self.data.roleset.isin(train_roleset)]
        data_test = self.data[self.data.roleset.isin(test_roleset)]
        
        self.train_mods = {}
        self.test_mods = {}
        
        self.metrics_test = {}
        self.metrics_train = {}
        
        for i in self.rolenums:
            self.train_mods[i] = {}
            self.test_mods[i] = {}

            self.metrics_train[i] = {}
            self.metrics_test[i] = {}
            
            self._fit(i, 0, data_train, data_test,
                      init_tolerance, train_tolerance, test_tolerance)
        

        
    def run_cv(self, folds=5, init_tolerance=1000., train_tolerance=1., test_tolerance=0.1):
        if init_tolerance is not None:
            self._train_initialization(init_tolerance)

        self.train_mods = {}
        self.test_mods = {}
        
        self.metrics_test = {}
        self.metrics_train = {}
        
        for i in self.rolenums:
            self.train_mods[i] = {}
            self.test_mods[i] = {}

            self.metrics_train[i] = {}
            self.metrics_test[i] = {}

            folder = LabelKFold(labels=self.data.sentenceid, n_folds=folds)
            
            for j, (train_index, test_index) in enumerate(folder):

                data_train = self.data.iloc[train_index]
                data_test = self.data.iloc[test_index]

                print
                print 'Iteration ', j
                print
                
                self._fit(i, j, data_train, data_test,
                          init_tolerance, train_tolerance, test_tolerance)
            
        return self

    def _train_initialization(self, init_tolerance):
        self.init_mods = {}
        
        for i in self.rolenums:
            print
            print 'Training initialization model ', i
            print

            self.init_mods[i] = Sprolim(data=self.data, k=i)
            self.init_mods[i].fit(tolerance=init_tolerance, verbose=True)

            print
            print 'Initialization metrics'
            print self.init_mods[i].metrics

    
    def _fit(self, i, j, data_train, data_test, init_tolerance, train_tolerance, test_tolerance):
        print 'Training model ', i
        print

        if init_tolerance is not None:
            self.train_mods[i][j] = Sprolim(data=data_train, k=i,
                                            initialized_model=self.init_mods[i])
        else:
            self.train_mods[i][j] = Sprolim(data=data_train, k=i)

        self.train_mods[i][j].fit(tolerance=train_tolerance, gd_init=False, verbose=True)

        print
        print 'Testing model ', i
        print

        self.test_mods[i][j] = Sprolim(data=data_test,
                                       initialized_model=self.train_mods[i][j],
                                       fix_params=True)
        self.test_mods[i][j].fit(tolerance=test_tolerance, verbose=True)

        self.metrics_train[i][j] = self.train_mods[i][j].metrics
        self.metrics_test[i][j] = self.test_mods[i][j].metrics

        print
        print 'Train metrics'
        print self.metrics_train[i][j]
        print
        print 'Test metrics'
        print self.metrics_test[i][j]
        print