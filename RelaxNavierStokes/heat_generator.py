import pickle
import subprocess
import time
import argparse
import heat_vis

if __name__=="__main__":

    #Build in argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--flags', type=str, default='_',
                        help = 'Any (options) flags which need' +
                        'to be passed to caller can be done so here')
    args, _ = parser.parse_known_args()
    
    
    #Parallel stuff
    MaxProcesses = 1
    Processes = []

    def checkrunning():
        for p in reversed(range(len(Processes))):
            if Processes[p].poll() is not None:
                del Processes[p]
        return len(Processes)
            
    #Fix some parameters
    N = 20
    dt = 0.001
    Mbase = 10

    #Define lists for those we want to vary
    Mrefs = [3,4]
    tdegrees = [0,1,2,3]
    sdegrees = [1,2,3]
    MPIProcesses = 8

    #Get the temporary file name unique to the set flags
    tmpname = args.flags.replace(' ', '_').replace('-','')
    
    #generate data
    for Mref in Mrefs:
        for tdegree in tdegrees:
            for sdegree in sdegrees:
                print('Mref =', Mref)
                print('tdegree =', tdegree)
                print('sdegree =', sdegree)
                
                process = subprocess.Popen('mpiexec -n %s python heat_caller.py %s --N %s --dt %s --tdegree %s --sdegree %s --Mbase %s --Mref %s --tmpname %s' % (MPIProcesses, '--'+args.flags, N, dt, tdegree, sdegree, Mbase, Mref, tmpname),
                                           shell=True, stdout=subprocess.PIPE)
                Processes.append(process)

                while checkrunning()==MaxProcesses:
                    time.sleep(1)
    while checkrunning()!=0:
        time.sleep(1)

    print('Data generation complete')
    
    #Pick up data and in list
    data = []
    for Mref in Mrefs:
        for tdegree in tdegrees:
            for sdegree in sdegrees:
                filename = 'tmp/heatdat%s%s%s%s%s%s%s%s' % (tmpname,
                                                            N, dt, tdegree,
                                                            sdegree, 1,
                                                            Mbase, Mref)
                try:
                    with open(filename, 'rb') as file:
                        out_ = pickle.load(file)
                        data_ = ['Mref %s, tdegree %s, sdegree %s' % (Mref,tdegree,sdegree),
                                 out_['dof'], out_['nnz'], out_['iterations'], out_['time_total'],out_['time_solve']]
                    data.append(data_)
                except Exception as e:
                    print('Loading %s failed with' % filename)
                    print("Error : "+str(e))
                        
    
    print(data)
    #Write to file
    file = open('heat%s.pickle' % tmpname, 'wb')
    pickle.dump(data,file)
    file.close()

    #Try and visualise
    try:
        heat_vis.visualise(tmpname)
    except Exception as e:
        print('Visualisation of %s failed with' % tmpname)
        print('Error : ' + str(e))
