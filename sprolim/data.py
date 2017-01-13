import numpy as np
import pandas as pd
import theano

from utility import *

from collections import defaultdict
from itertools import permutations,  combinations, combinations_with_replacement

import warnings
warnings.simplefilter(action = "ignore")

class SprolimData(object):
    '''
    wrapper for pandas DataFrame

    creates theano shared variables for various columns in data and makes them
    available as attributes in the same way a standard DataFrame does; converts
    columns to integer indices prior to wrapping in shared variable
    '''
    
    def __init__(self, data, nltrs=None):
        '''
        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe containing the following columns: 
            - sentenceid
            - predicate
            - predpos
            - gramfunc
            - predpos
            - argpos
            - property
            - applicable
            - response

        nltrs : pandas.DataFrame
            A dataframe containing the following columns:
            - predicate
            - nltrs
        '''
        
        self._data = data
        self._nltrs = nltrs
        
        self._ident = ''.join([str(i) for i in np.random.choice(9, size=10)])

        self._extract_data_attrs()
        self._extract_predicate_list()
        self._append_argument_info()
        self._append_combined_columns()
        self._indexify_response()
        self._partition_data()
        self._create_predicate_token_indices()
        self._create_shared_variables()
        self._set_counts()

    def _extract_data_attrs(self):
        self._nproperties = self._data.property.unique().shape[0]
        self._ngramfuncs = get_num_of_types(self._data.gramfunc)

    def _extract_predicate_list(self):
        predicates = self._data.predicate.astype('category')
        self._predicate_list = np.array(predicates.cat.categories)

        self._ycats = np.array(self._data.gramfunc.unique())
        
    def _append_argument_info(self):

        argcounter = lambda x: np.unique(x).shape[0]

        h = ['sentenceid', 'predicate', 'predpos']

        ## append the number of arguments for each predicate token and the relative
        ## position of each argument relative to the predicate
        self._data['nargs'] = self._data.groupby(h).argpos.transform(argcounter)
        self._data['argposrel'] = self._data.groupby(h).argpos.transform(get_codes)

        ## append the number of l-thematic roles for each predicate
        if self._nltrs is None:
            self._nltrs = self._data.groupby('predicate').nargs.apply(np.max).reset_index()
            self._data['nltrs'] = self._data.groupby('predicate').nargs.transform(np.max)
        else:
            self._data = pd.merge(self._data, self._nltrs)

        ## append dummy variables for syntactic position
        self._data['indicator'] = 1
        syntax = self._data.pivot_table(index=h, columns='gramfunc',
                                        values='indicator',
                                        aggfunc=lambda x: sum(x)/self._nproperties,
                                        fill_value=0.).reset_index()

        self._data = pd.merge(self._data, syntax)

    def _append_combined_columns(self):
        self._data['predicatearg'] = self._data.predicate + ':' + self._data.argpos.astype(str)
        self._data['sentpredicate'] = self._data.sentenceid + ':' + self._data.predicate
        self._data['sentpredicatearg'] = self._data.sentpredicate + ':' + self._data.argpos.astype(str)

    def _indexify_response(self):
        self._data['response'] = self._data.response-self._data.response.min()

    def _partition_data(self):
        partition = {i: self._data[self._data.nltrs==i] for i in self._data.nltrs.unique()}
        
        self._data_partitioned = {i: {j: cell[cell.nargs==j] for j in cell.nargs.unique()}
                                  for i, cell in partition.iteritems()}

    def _create_predicate_token_indices(self):
        self._predicatetokenindices = defaultdict(dict)
        
        for i, outer_partition in self._data_partitioned.iteritems():
            for j, inner_partition in outer_partition.iteritems():
                unique_sentpredicates = inner_partition.sentpredicate.astype('category').cat.categories
                predicate_tokens = unique_sentpredicates.map(lambda x: x.split(':')[1])
            
                self._predicatetokenindices[i][j] = np.array(get_codes(pd.Series(predicate_tokens)))

    def _wrap_in_shared(self, i, j, colname, cats=None):
        return theano.shared(np.array(get_codes(self._data_partitioned[i][j][colname], cats)),
                             name=colname+str(i)+str(j)+self._ident)
            
    def _create_shared_variables(self):

        heads = ['sentenceid', 'predicate', 'argposrel', 'predicatearg',
                 'sentpredicate', 'sentpredicatearg', 'property', 'gramfunc',
                 'applicable', 'response']

        self.gramfunc = defaultdict(dict)
        
        for h in heads:
            self.__dict__[h] = defaultdict(dict)
            
            for i, outer_partition in self._data_partitioned.iteritems():
                for j, inner_partition in outer_partition.iteritems():

                    self.gramfunc[i][j] = theano.shared(np.array(inner_partition[self._ycats]),
                                                                 name='syntax'+str(i)+str(j)+self._ident)

                    if h=='predicate':
                        self.__dict__[h][i][j] = self._wrap_in_shared(i, j, h, self._predicate_list)
                    else:
                        self.__dict__[h][i][j] = self._wrap_in_shared(i, j, h)

    def _set_counts(self):
        self.npredicates_total = self._predicate_list.shape[0]

        num = lambda x: len(x.unique())
        
        nsents_per_predicate = self._data.groupby(['predicate']).sentenceid.apply(num)
        self.nsents_per_predicate = theano.shared(np.array(nsents_per_predicate),
                                                  name='nsentsperpredicate'+self._ident)

        self.nsentences = defaultdict(dict)
        self.nproperties = defaultdict(dict)
        self.npredicates = defaultdict(dict)
        self.npredicatetokens = defaultdict(dict)
        self.nargtokens = defaultdict(dict)

        for i, outer_partition in self._data_partitioned.iteritems():
            for j, inner_partition in outer_partition.iteritems():
                d = self._data_partitioned[i][j]
                
                self.nsentences[i][j] = get_num_of_types(d.sentenceid)
                self.nproperties[i][j] = get_num_of_types(d.property)
                self.npredicates[i][j] = get_num_of_types(d.predicate)
                self.npredicatetokens[i][j] = get_num_of_types(d.sentpredicate)
                self.nargtokens[i][j] = get_num_of_types(d.sentpredicatearg)

    @property
    def data(self):
        return self._data_partitioned

    @property
    def ngramfuncs(self):
        return self._ngramfuncs

    
if __name__ == '__main__':
    d = pd.read_csv('../bin/data.tmp')
    
    # d = pd.concat([pd.read_csv('protoroles_eng_pb_08302015.tsv', sep='\t'),
    #                pd.read_csv('ud_info.tsv', sep='\t')],
    #               axis=1)

    # d = d.rename(columns=lambda x: x.replace('.', '').lower())

    # d['response'] = d['response'].astype(int)
    # d['applicable'] = d['applicable'].astype(int)
    # d['argposud'] = d['argposud'].map(lambda x: x.split(',')[0]).astype(int)

    d = d[['sentenceid', 'roleset', 'predtokenud', 'argposud', 'gramfuncud',
           'property', 'response', 'applicable']]
    
    d = d.rename(columns={'roleset': 'predicate',
                          'argposud': 'argpos',
                          'predtokenud': 'predpos',
                          'gramfuncud': 'gramfunc'})
    
    sd = SprolimData(d)
