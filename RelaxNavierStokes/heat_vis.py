import pickle
import pandas as pd
import os

def visualise(name):
    if os.path.isdir('data')==False:
        os.mkdir('data')
    
    df = pd.DataFrame()

    with open('heat%s.pickle' % name, 'rb') as file:
        data = pickle.load(file)


    df = pd.DataFrame.from_dict({'Run': [],
                                 'DOF': [],
                                 'NNZ': [],
                                 'Iterations': [],
                                 'Total time': [],
                                 'Solve time': []})

    for i in range(len(data)):
        df.loc[len(df)] = data[i]

    #Save to file
    filename = 'data/heat' + name
    df.to_csv(filename+'.csv')
    with open(filename+'.tex','w') as f:
        print(df.to_latex(),file=f)

    print(df)

    return None


if __name__=="__main__":
    visualise('_')
