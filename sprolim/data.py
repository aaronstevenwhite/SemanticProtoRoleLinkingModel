import numpy as np
import pandas as pd
import theano

from utility import *

from itertools import permutations,  combinations, combinations_with_replacement

import warnings
warnings.simplefilter(action = "ignore")

class Data(object):
    '''
    wrapper for pandas DataFrame

    creates theano shared variables for various columns in data and makes them
    available as attributes in the same way a standard DataFrame does; converts
    columns to integer indices prior to wrapping in shared variable
    '''
    
    def __init__(self, data, k, ycats):
        self._data = data
        self.k = k
        self.ycats = ycats
        self.ident = ''.join([str(i) for i in np.random.choice(9, size=10)])

        self._extract_data_attrs()
        self._create_combined_columns()
        self._partition_data()
        self._create_roleset_token_indices()
        self._create_shared_variables()
        self._initialize_maps()
        self._set_counts()

    def _extract_data_attrs(self):
        self.max_ltrs = self._data.argnum.max()
        self.unique_argnums = self._data.argnum.unique()
        self.rolesets_list = np.array(self._data.roleset.astype('category').cat.categories)

    def _partition_data(self):
        self._data_partitioned = {j : self._data[self._data.argnum==j] for j in self.unique_argnums}
        
    def _wrap_in_shared(self, j, colname, cats=None):
        return theano.shared(np.array(get_codes(self._data_partitioned[j][colname], cats)),
                             name=colname+str(j)+self.ident)

    def _create_combined_columns(self):
        self._data['rolesetarg'] = self._data.roleset + ':' + self._data.argpos.astype(str)
        self._data['sentroleset'] = self._data.sentenceid + ':' + self._data.roleset
        self._data['sentrolesetarg'] = self._data.sentroleset + ':' + self._data.argpos.astype(str)
        
        self._data['response'] = self._data.response-1


    def _create_roleset_token_indices(self):
        self.rolesettokenindices = {}
        
        for j in self.unique_argnums:
            unique_sentrolesets = self._data_partitioned[j].sentroleset.astype('category').cat.categories
            roleset_tokens = unique_sentrolesets.map(lambda x: x.split(':')[1])
            
            self.rolesettokenindices[j] = np.array(get_codes(pd.Series(roleset_tokens)))
        
    def _create_shared_variables(self):

        heads = {'sentenceid': [],
                 'roleset': self.rolesets_list,
                 'argposrel': [],
                 'rolesetarg': [],
                 'sentroleset': [],
                 'sentrolesetarg': [],
                 'property': [],
                 'gramfuncud': self.ycats,
                 'applicable': [],
                 'response': []}

        self.other_positions = {}
        
        for h, cats in heads.items():
            self.__dict__[h] = {}
            
            for j in self.unique_argnums:
                self.other_positions[j] = theano.shared(np.array(self._data_partitioned[j][self.ycats]),
                                                        name='other_positions'+str(j)+self.ident)

                if len(cats):
                    self.__dict__[h][j] = self._wrap_in_shared(j, h, cats)
                else:
                    self.__dict__[h][j] = self._wrap_in_shared(j, h)

            
    def _initialize_maps(self):
        self.ltr_perms = {}
        self.position_role_map = {}

        self.num_of_ltr_perms = {}

        role_combs = list(combinations_with_replacement(range(self.k), self.max_ltrs))
        self.role_perms = np.array(list({tuple(perm) for comb in role_combs for perm in permutations(comb)}))
        
        for j in self.unique_argnums:
            ltr_perms = np.array(list({tuple(perm[:j]) for perm in np.array(list(permutations(range(self.max_ltrs))))}))
            self.ltr_perms[j] = ltr_perms[np.lexsort(np.transpose(ltr_perms)[::-1])]
            
            self.position_role_map[j] = theano.shared(np.transpose(self.role_perms[:,self.ltr_perms[j]], (2,0,1)),
                                                   name='position_role_map'+str(j)+self.ident)

            self.num_of_ltr_perms[j] = self.ltr_perms[j].shape[0]
            
    def _set_counts(self):
        self.num_of_rolesets_total = self.rolesets_list.shape[0]

        num = lambda x: len(x.unique())
        
        num_of_sents_per_roleset = self._data.groupby(['roleset']).sentenceid.apply(num)
        self.num_of_sents_per_roleset = theano.shared(np.array(num_of_sents_per_roleset),
                                                      name='nsentsperroleset'+self.ident)

        self.num_of_sentences = {}
        self.num_of_properties = {}
        self.num_of_rolesets = {}
        self.num_of_rolesettokens = {}
        self.num_of_argtokens = {}
        
        for j in self.unique_argnums:
            self.num_of_sentences[j] = get_num_of_types(self._data_partitioned[j].sentenceid)
            self.num_of_properties[j] = get_num_of_types(self._data_partitioned[j].property)
            self.num_of_rolesets[j] = get_num_of_types(self._data_partitioned[j].roleset)
            self.num_of_rolesettokens[j] = get_num_of_types(self._data_partitioned[j].rolesetarg)
            self.num_of_argtokens[j] = get_num_of_types(self._data_partitioned[j].sentrolesetarg)

    @property
    def data(self):
        return self._data_partitioned
