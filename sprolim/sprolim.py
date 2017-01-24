import sys
import cPickle as pickle
import numpy as np
import scipy as sp
import pandas as pd
import theano
import theano.tensor as T

from collections import defaultdict
from itertools import combinations, combinations_with_replacement, permutations
from scipy.misc import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.cross_validation import LabelKFold, train_test_split

from utility import *
# from data import SprolimData

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action = "ignore")

#theano.config.profile = True

class Sprolim(object):
    
    def __init__(self, nprotoroles=2, nverbdims=None, nverbsenses=None, orthogonalize=True):
        '''
        Parameters
        ----------
        nprotoroles : int
            the number of protoroles the model assumes
        nverbdims : int or NoneType
            the number of dimensions for the verb embeddings
        nverbsenses : int or NoneType
            the number of verb embeddings per verb type
        orthoganlize : bool
            whether to orthogonalize the verb embeddings
        '''

        self.nprotoroles = nprotoroles
        self.nverbdims = nverbdims
        self.orthoganlize = orthogonalize
        self.nverbsenses = nverbsenses if nverbsenses is not None else 1
        
        self._ident = ''.join([str(i) for i in np.random.choice(9, size=10)])
        
    def _initialize_model(self):

        self._initialize_maps()
        
        self._representations = {}
        
        self._initialize_mixture()
        self._initialize_canonicalization()
        self._initialize_applicability()
        self._initialize_rating()
        self._initialize_syntax()

        self._initialize_loss()
        
    def _initialize_maps(self):
        self.role_perms = {}
        self.ltr_perms = defaultdict(dict)
        self.position_role_map = defaultdict(dict)

        self.nltrperms = defaultdict(dict)

        for i, outer_partition in self.data.iteritems():
            role_combs = list(combinations_with_replacement(range(self.nprotoroles), i))
            self.role_perms[i] = np.array(list({tuple(perm) for comb in role_combs for perm in permutations(comb)}))

            for j, inner_partition in outer_partition.iteritems():
                ltr_perms = np.array(list({tuple(perm[:j]) for perm in np.array(list(permutations(range(i))))}))
                self.ltr_perms[i][j] = ltr_perms[np.lexsort(np.transpose(ltr_perms)[::-1])]

                self.position_role_map[i][j] = theano.shared(np.transpose(self.role_perms[i][:,self.ltr_perms[i][j]], (2,0,1)),
                                                             name='position_role_map'+str(i)+str(j)+self._ident)

                self.nltrperms[i][j] = self.ltr_perms[i][j].shape[0]

        
    def _initialize_mixture(self):
        role_ll = defaultdict(dict)

        if self.nverbdims is not None:
            verbreps = np.random.normal(size=[self.data.npredicates_total, self.nverbdims])
            self._representations['verbreps'] = theano.shared(verbreps, name='verbreps'+self._ident)
        
        for i, outer_partition in self.data.predicate.iteritems():
            if self.nverbdims is None:
                mixture_aux = np.zeros([self.data.npredicates_total, i, self.nprotoroles])

                self._representations['mixture_'+str(i)] = theano.shared(mixture_aux,
                                                                         name='mixture'+str(i)+self._ident)

                mixraw = self._representations['mixture_'+str(i)]
            else:
                map_to_protorole = np.random.normal(size=[self.nverbdims, i, self.nprotoroles])

                self._representations['map_to_protorole_'+str(i)] = theano.shared(map_to_protorole,
                                                                         name='map_to_protorole'+str(i)+self._ident)

                mixraw = T.tensordot(T.tanh(self._representations['verbreps']),
                #mixraw = T.tensordot(self._representations['verbreps'],
                                     self._representations['map_to_protorole_'+str(i)],
                                     axes=1)

            mixture_exp = T.exp(mixraw)
            mixture = mixture_exp / T.sum(mixture_exp, axis=2)[:,:,None]

            log_mixture = T.log(T.swapaxes(mixture, 0, 2))

            role_perms_logprob = T.sum(log_mixture[self.role_perms[i],T.arange(i)[None,:]], axis=1)
            
            for j, inner_partition in outer_partition.iteritems():
                log_divisor = T.log(self.data.nsents_per_predicate[inner_partition])
                role_ll[i][j] = T.transpose(role_perms_logprob)[inner_partition] -\
                                log_divisor[:,None]
                
        self._role_ll = role_ll
        self._role_prior = 0. #(self.alpha - 1.)*T.sum(log_mixture)

    def _initialize_canonicalization(self):
        canon_ll = defaultdict(dict)
        
        for i, outer_partition in self.data.predicatetokenindices.iteritems():
            for j, inner_partition in outer_partition.iteritems():
                # if self.nverbdims is None:
                canonicalization_prob_aux = np.tile(1./np.arange(1, self.nltrperms[i][j]+1),
                                                    [self.data.npredicates[i][j], 1])

                self._representations['canonicalization_'+str(i)+str(j)] = theano.shared(canonicalization_prob_aux,
                                                                                         name='canonicalization_prob'+\
                                                                                              str(i)+str(j)+self._ident)

                canonicalization_exp = T.exp(self._representations['canonicalization_'+str(i)+str(j)])
                canonicalization_prob = canonicalization_exp / T.sum(canonicalization_exp, axis=1)[:,None]
                canonicalization_logprob = T.log(canonicalization_prob)[inner_partition]

                # else:
                #     canonicalization_map = np.random.normal(size=[self.nverbdims, self.nltrperms[i][j]+1])

                #     self._representations['canonicalization_map_'+str(i)+str(j)] = theano.shared(canonicalization_map,
                #                                                                              name='canonicalization_map'+\
                #                                                                                   str(i)+str(j)+self._ident)

                #     canonraw = T.dot(T.tanh(self._representations['verbreps']),
                #                      self._representations['canonicalization_map_'+str(i)+str(j)])
                    

                #     canonicalization_exp = T.exp(canonraw)
                #     canonicalization_prob = canonicalization_exp / T.sum(canonicalization_exp, axis=1)[:,None]
                #     canonicalization_logprob = T.log(canonicalization_prob)[self.data.predicatetypeindices[i][j]]
                    
                canon_ll[i][j] = canonicalization_logprob[self.data.sentpredicate[i][j],None,:] - T.log(j)
            
        self._canonicalization_ll = canon_ll

    def _initialize_applicability(self):
        # if self.initialized_model is None:
        #     #appl_aux = np.random.normal(0, 1, size=[self.nprotoroles, self.num_of_properties, 2])
        #     appl_aux = np.zeros([self.nprotoroles, self.num_of_properties, 2])
        # else:
        #     appl_aux = self.initialized_model.representations['applicable'].eval()

        appl_aux = np.zeros([self.nprotoroles, self.data.nproperties, 2])
        
        self._representations['applicable'] = theano.shared(appl_aux, name='appl'+self._ident)

        weights_appl_aux = np.zeros([self.data.nannotators, 2])

        self._representations['appl_weights'] = theano.shared(weights_appl_aux, name='appl_weights'+self._ident)
        
        appl_uprob = T.exp(self._representations['applicable'][None,:,:,:] +\
                           self._representations['appl_weights'][:,None,None,:])
        appl_prob = appl_uprob / T.sum(appl_uprob, axis=3)[:,:,:,None]

        appl_ll = defaultdict(dict)
        
        for i, outer_partition in self.data.iteritems():
            for j, inner_partition in outer_partition.iteritems():
                appl_ll[i][j] = T.log(appl_prob[self.data.annotatorid[i][j][:,None,None],
                                                self.position_role_map[i][j][self.data.argposrel[i][j]],
                                                self.data.property[i][j][:,None,None],
                                                self.data.applicable[i][j][:,None,None]])

        self._appl_ll = appl_ll

    def _initialize_rating(self):
        # if self.initialized_model is None:
        #     #rating_aux = np.random.normal(0, 1, size=[self.nprotoroles, self.num_of_properties])
        #     rating_aux = np.zeros([self.nprotoroles, self.num_of_properties])
        # else:
        #     rating_aux = self.initialized_model.representations['rating'].eval()

        rating_aux = np.zeros([self.nprotoroles, self.data.nproperties])
            
        self._representations['rating'] = theano.shared(rating_aux, name='rating'+self._ident)

        # if self.initialized_model is None:
        #     jumps_aux = np.array([-np.inf] + [1]*(np.max(self.data.response)-1) + [-np.inf])
        # else:
        #     jumps_aux = self.initialized_model.representations['jumps'].eval()

        weights_add = np.zeros([self.data.nannotators, self.data.nresponsetypes])
        #weights_mult = np.zeros([self.data.nannotators, np.max(self.data.response)+1])
        
        self._representations['rating_weights_add'] = theano.shared(weights_add, name='rating_weights_add'+self._ident)
        #self._representations['rating_weights_mlt'] = theano.shared(weights, name='rating_weights_mult'+self._ident)

        #wmcum = T.extra_ops.cumsum(T.square(self._representations['rating_weights_mlt']), axis=1)[:,None,None,:]
        wmcum = (np.arange(self.data.nresponsetypes)+1.)[None,None,None,:]
        
        rate_uprob = T.exp(wmcum*self._representations['rating'][None,:,:,None] +\
                           self._representations['rating_weights_add'][:,None,None,:])
        rate_prob = rate_uprob / T.sum(rate_uprob, axis=3)[:,:,:,None]
        
        rate_ll = defaultdict(dict)

        for i, outer_partition in self.data.iteritems():
            for j, inner_partition in outer_partition.iteritems():
                rate_ll[i][j] = T.log(rate_prob[self.data.annotatorid[i][j][:,None,None],
                                                self.position_role_map[i][j][self.data.argposrel[i][j]],
                                                self.data.property[i][j][:,None,None],
                                                self.data.response[i][j][:,None,None]])

        self._rating_ll = rate_ll
    
    def _initialize_syntax(self):
        # if self.initialized_model is None:
        #     role_synt_aux = np.random.normal(0., 1., size=[self.num_of_syntpos,self.nprotoroles])
        #     synt_synt_aux = LogisticRegression(fit_intercept=False,
        #                            multi_class='multinomial',
        #                            solver='newton-cg').fit(self.other_positions,
        #                                                    self.data.gramfuncud).coef_
        # 
        # else:
        #     role_synt_aux = self.initialized_model.representations['role_syntax'].eval()
        #     synt_synt_aux = self.initialized_model.representations['syntax_syntax'].eval()

        role_synt_aux = np.random.normal(0., 1., size=[self.data.ngramfuncs, self.nprotoroles])
        synt_synt_aux = LogisticRegression(fit_intercept=False,
                                           multi_class='multinomial',
                                           solver='newton-cg').fit(self.data.rawdata[self.data.levels('gramfunc')],
                                                                   self.data.rawdata.gramfunc).coef_
  
        self._representations['role_syntax'] = theano.shared(role_synt_aux, name='role_synt'+self._ident)
        role_synt = self._representations['role_syntax']

        self._representations['syntax_syntax'] = theano.shared(synt_synt_aux, name='synt_synt'+self._ident)
        synt_synt = self._representations['syntax_syntax']

        synt_ll = defaultdict(dict)
        
        for i, outer_partition in self.data.iteritems():            
            for j, inner_partition in outer_partition.iteritems():                
                synt_synt_sum = T.sum(self.data.gramfunc_global[i][j][:,None,:]*synt_synt[None,:,:], axis=2)
                synt_potential = synt_synt_sum[:,:,None]+role_synt[None,:,:]

                synt_potential_exp = T.exp(synt_potential)
                synt_prob = synt_potential_exp / T.sum(synt_potential_exp, axis=1)[:,None,:]
                

                synt_ll[i][j] = T.log(synt_prob[T.arange(inner_partition.shape[0])[:,None,None],
                                                self.data.gramfunc[i][j][:,None,None],
                                                self.position_role_map[i][j][self.data.argposrel[i][j]]])
                synt_ll[i][j] = synt_ll[i][j] - T.log(float(self.data.nproperties))
 
        self._syntax_ll = synt_ll

    def _initialize_loss(self):
        self._total_loglike_sum = 0.
        self._total_loglike_sum_synt_only = 0.

        self._role_perm = defaultdict(dict)
        
        for i, outer_partition in self.data.iteritems():
            for j, inner_partition in outer_partition.iteritems():
                ## likelihood
                inner_sum = self._canonicalization_ll[i][j]+\
                            self._appl_ll[i][j]+\
                            self._rating_ll[i][j]
                
                # if not self.semantics_only:
                #     inner_sum += self._syntax_ll[i][j]
                inner_sum += self._syntax_ll[i][j]

                self._role_perm[i][j] = self._role_ll[i][j][:,:,None] + inner_sum
                    
                perm_ll = T.log(T.sum(T.exp(inner_sum), axis=2))
                
                self._total_loglike_sum += T.sum(T.log(T.sum(T.exp(self._role_ll[i][j] +\
                                                                   perm_ll),
                                                            axis=1)))

                inner_sum_synt_only = self._canonicalization_ll[i][j]+self._syntax_ll[i][j]
                perm_ll_synt_only = T.log(T.sum(T.exp(inner_sum_synt_only), axis=2))

                self._total_loglike_sum_synt_only += T.sum(T.log(T.sum(T.exp(self._role_ll[i][j] +\
                                                                            perm_ll_synt_only),
                                                                      axis=1)))
                
        self._total_loglike_sum += self._role_prior

        if self.orthoganlize:
            verbrep2 = T.dot(T.tanh(self._representations['verbreps']).T, T.tanh(self._representations['verbreps']))
            verbrep2_rawsum = T.sum(T.square(verbrep2 - verbrep2*T.identity_like(verbrep2)))
            self._total_loglike_sum += -1e7*verbrep2_rawsum/(self.nverbdims**2*self.data.npredicates_total**2)

    def _initialize_updaters(self, fix_params):
        update_dict_gd = []
        update_dict_ada = []

        self.rep_grad_hist_t = {}
        
        for name, rep in self._representations.items():            
            if not fix_params or name.split('_')[0] in ['canonicalization', 'mixture']:
                rep_grad = T.grad(self._total_loglike_sum, rep)

                # if self.initialized_model is None or name.split('_')[0] in ['canonicalization', 'mixture']:
                self.rep_grad_hist_t[name] = theano.shared(np.ones(rep.shape.eval()), name=name+'_hist'+self._ident)
                # else:
                #     self.rep_grad_hist_t[name] = theano.shared(self.initialized_model.rep_grad_hist_t[name].eval(),
                #                                                name=name+'_hist'+self._ident)

                rep_grad_adj = rep_grad / (T.sqrt(self.rep_grad_hist_t[name]))
                
                # if name == 'jumps':
                #     learning_rate = 0.000001
                # else:
                learning_rate = 0.00001

                update_dict_gd += [(self.rep_grad_hist_t[name], self.rep_grad_hist_t[name] + T.power(rep_grad, 2)),
                                    (rep, rep + learning_rate*rep_grad)]
                update_dict_ada += [(self.rep_grad_hist_t[name], self.rep_grad_hist_t[name] + T.power(rep_grad, 2)),
                                    (rep, rep + rep_grad_adj)]  
            
        self.updater_gd = theano.function(inputs=[],
                                          outputs=[self._total_loglike_sum,
                                                   self._total_loglike_sum_synt_only],
                                          updates=update_dict_gd,
                                          name='updater_gd_'+self._ident)

        self.updater_ada = theano.function(inputs=[],
                                           outputs=[self._total_loglike_sum,
                                                    self._total_loglike_sum_synt_only],
                                           updates=update_dict_ada,
                                           name='updater_ada'+self._ident)


    def fit(self, data, iterations=1000, tolerance=-np.inf, gd_init=20, fix_params=False, verbose=False):
        '''
        Parameters
        ----------
        data : sprolim.SprolimData
            Wrapped SPR1 or SPR2 data

        iterations : int        
        tolerance : float
        gd_init : int
        fix_params : bool
        verbose : bool
        '''

        self.data = data
        
        self._initialize_model()
        
        self._initialize_updaters(fix_params)
        
        prev_ll = -np.inf
        
        for i in range(iterations):
            if i < gd_init:
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
            num_params = [np.prod(rep.shape.eval()) for name, rep in self._representations.items()
                          if name not in ['applicable', 'rating', 'jumps']]
            
            return -2*self.ll_synt_only, -2*self.ll_synt_only + 2*np.sum(num_params)

        else:
            num_params = [np.prod(rep.shape.eval()) for rep in self._representations.values()]

            return -2*self.ll, -2*self.ll + 2*np.sum(num_params)
    
    @property
    def mixture(self):
        mixture_dict = {}
        
        for i, data in self.data_split.iteritems():
            mixture_exp = T.exp(self._representations['mixture_'+str(i)])
            mixture = (mixture_exp / T.sum(mixture_exp, axis=2)[:,:,None]).eval()

            mixture_dict[i] = pd.Panel(mixture,
                                          items=data.rolesets_list,
                                          major_axis=['ltr'+str(j) for j in range(data.max_ltrs)],
                                          minor_axis=['role'+str(j) for j in range(self.nprotoroles)])

        return mixture_dict
            
    @property
    def canonicalization(self):
        canon_dict = {}
        
        for i, data in self.data_split.iteritems():
            canon_dict[i] = {}
            for j in data.unique_argnums:
                canon_exp = T.exp(self._representations['canonicalization_'+str(i)+str(j)])
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
                                               minor_axis=['role'+str(k) for k in range(self.nprotoroles)])
                                               

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
        return pd.DataFrame(np.transpose(self._representations['rating'].eval()),
                            index=self.data.levels('property'))

    @property
    def applicable(self):
        appl_exp = T.exp(self._representations['applicable'])
        appl = appl_exp / T.sum(appl_exp, axis=2)[:,:,None]
        
        return pd.DataFrame(np.transpose(appl[:,:,1].eval()),
                            index=self.data.levels('property'))

    @property
    def role_syntax(self):
        return pd.DataFrame(self._representations['role_syntax'].eval(),
                            columns=['role'+str(i) for i in range(self.nprotoroles)],
                            index=self.data.levels('gramfunc'))

    @property
    def syntax_syntax(self):
        return pd.DataFrame(self._representations['syntax_syntax'].eval(),
                            index=self.data.levels('gramfunc'),
                            columns=self.data.levels('gramfunc'))

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
        params_Xy = self._representations['role_syntax'].eval()
        params_yy = self._representations['syntax_syntax'].eval()

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

                    mixture_maxed = np.eye(self.nprotoroles)[np.argmax(mixture_selected, axis=2)]
                    
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

        self.rating.to_csv(name('rating')+str(self.nprotoroles)+'.csv')
        self.applicable.to_csv(name('applicable')+str(self.nprotoroles)+'.csv')
        self.syntax_syntax.to_csv(name('syntax_syntax')+str(self.nprotoroles)+'.csv')
        self.role_syntax.to_csv(name('role_syntax')+str(self.nprotoroles)+'.csv')
        self.mixture_canonicalized.to_csv(name('mixture_canonicalized')+str(self.nprotoroles)+'.csv')
        self.metrics.to_csv(name('metrics')+str(self.nprotoroles)+'.csv')
        self.prediction.to_csv(name('prediction')+str(self.nprotoroles)+'.csv')
        self.confusion.to_csv(name('confusion')+str(self.nprotoroles)+'.csv')

if __name__ == '__main__':
    from data import main, SprolimData
    sd = main()

    m = Sprolim(nverbdims=2)
    m.fit(sd, verbose=True)
