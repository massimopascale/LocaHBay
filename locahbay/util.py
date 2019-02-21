''' utility functions 
'''
import os 

def dat_dir():  
    ''' local directory for dumping files. This is mainly used for  
    the test runs.
    '''
    if os.environ.get('LOCAHBAY_DIR') is None: 
        raise ValueError("set $LOCAHBAY_DIR environment varaible!") 
    return os.environ.get('LOCAHBAY_DIR') 


def code_dir(): 
    ''' location of the repo directory. set $PYSPEC_CODEDIR 
    environment varaible in your bashrc file. 
    '''
    if os.environ.get('LOCAHBAY_CODEDIR') is None: 
        raise ValueError("set $LOCAHBAY_CODEDIR environment varaible!") 
    return os.environ.get('LOCAHBAY_CODEDIR') 
    

def fig_dir(): 
    ''' directory to dump all the figure files 
    '''
    figdir = os.path.join(code_dir(), 'figs')
    if os.path.isdir(figdir):
        return figdir
    else: 
        raise ValueError("create figs/ folder in $LOCAHBAY_CODEDIR directory for figures")


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    docdir = os.path.join(code_dir(), 'doc')
    if os.path.isdir(docdir):
        return docdir
    else: 
        raise ValueError("create doc/ folder in $LOCAHBAY_CODEDIR directory for documntation")

