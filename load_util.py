import pickle

def load_st(model,path):
    if path is None:
        return model
    with open(path,'rb') as fp:
        state_dictionary = pickle.load(fp)
    
    model.load_state_dict(state_dictionary)
    return model
