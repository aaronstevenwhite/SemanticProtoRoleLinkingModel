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
