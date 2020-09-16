    if "get_ipython" in locals():
        %config Testing.update=False
    
# compositions for setting items of mappings
    
    f = a[{'a': 1, 'b': 2}]
    f['c'] = Self['a']
    f['a'] = Self['b']*4
    
    >>> f()
    {'a': 8, 'b': 2, 'c': 1}
    
    
    df = Pose + 'pandas.util.testing.makeDataFrame' + ...


    >>> df.columns.tolist()
    ['A', 'B', 'C', 'D']
    
## Setting attributes
    
    g = Pose()
    g.columns = Self.columns().map(str.lower)
    
    >>> g(df).columns.tolist()
    ['a', 'b', 'c', 'd']