def get_codes(x, cats=None):
    if cats is None:
        return x.astype('category').cat.codes
    else:
        return x.astype('category', categories=cats).cat.codes

def get_num_of_types(x):
    return x.unique().shape[0]
