import pickle
import subprocess
import time
import argparse
import chorin_vis

if __name__=="__main__":

    #Build in argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--flags', type=str, default='_',
                        help = 'Any (optional) flag which needs ' +
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
    N_ = 10
    dt_ = 0.01
    Mbase = 10

    #Define lists for those we want to vary
    Mrefs = [3]
    tdegrees = [0,1]
    sdegrees = [1,2]
    Rs = [1e1]
    MPIProcesses = 8
    dts = [0.01,0.005,0.0025] #List should contain dt_

    #Make temporary file name unique under flags
    tmpname = args.flags.replace(' ', '_').replace('-','')
    
    #generate data
    for Mref in Mrefs:
        for tdegree in tdegrees:
            for sdegree in sdegrees:
                for R in Rs:
                    for dt in dts:
                        N = int(N_ * (dt_/dt))
                        print('Mref =', Mref)
                        print('tdegree =', tdegree)
                        print('sdegree =', sdegree)
                        print('R =', R)
                        print('dt =', dt)
                        print('N =', N)

                        process = subprocess.Popen('mpiexec -n %s python chorin_caller.py %s --N %s --dt %s --tdegree %s --sdegree %s --R %s --Mbase %s --Mref %s --tmpname %s' % (MPIProcesses, '--'+args.flags, N, dt, tdegree, sdegree, R, Mbase, Mref, tmpname),
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
                for R in Rs:
                    for dt in dts:
                        N = int(N_ * (dt_/dt))
                        filename = 'tmp/chorindat%s%s%s%s%s%s%s%s%s' % (tmpname, N, dt,
                                                                        tdegree, sdegree, R,
                                                                        1, Mbase, Mref)
                        try:
                            with open(filename, 'rb') as file:
                                out_ = pickle.load(file)
                                data_ = ['Mref %s, tdegree %s, sdegree %s, R %s, dt %s' % (Mref,tdegree,sdegree,R,dt),
                                         out_['dof'],out_['nnz'],
                                         out_['error'], out_['iterations'], out_['newton iterations'],
                                         out_['time_total'],out_['time_solve']]
                            data.append(data_)
                        except Exception as e:
                            print('Loading %s failed with' % filename)
                            print("Error : "+str(e))
                        
    
    print(data)
    #Write to file
    file = open('chorin%s.pickle' % tmpname, 'wb')
    pickle.dump(data,file)
    file.close()
    
    #Try and visual
    try:
        chorin_vis.visualise(tmpname)
    except Exception as e:
        print('Visualisation of %s failed with' % tmpname)
        print('Error : ', str(e))
