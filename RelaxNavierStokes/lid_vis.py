import pickle
import pandas as pd
import os

def visualise(name,parallel=False):
    if os.path.isdir('data')==False:
        os.mkdir('data')
    
    df = pd.DataFrame()

    if parallel:
        readname = 'lidParallel%s.pickle' % name
    else:
        readname = 'lid%s.pickle' % name

    with open(readname, 'rb') as file:
        data = pickle.load(file)


    df = pd.DataFrame.from_dict({'Run': [],
                                 'DOF': [],
                                 'NNZ': [],
                                 'MG iterations': [],
                                 'Newton iterations': [],
                                 'Total time': [],
                                 'Solve time': []})

    for i in range(len(data)):
        df.loc[len(df)] = data[i]

    #Save to file
    filename = 'data/lidParallel' + name
    df.to_csv(filename+'.csv')
    with open(filename+'.tex','w') as f:
        print(df.to_latex(),file=f)

    print(df)

    return None


if __name__=="__main__":
    visualise('_',parallel=True)
    visualise('stepper',parallel=True)
