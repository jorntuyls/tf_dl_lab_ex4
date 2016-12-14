
import numpy as np
import time
import math

class Master:

    def __init__(self):
        self.worker_file_list = []

    def get_random_hyperparameter_configuration(self):
        x = np.random.uniform(0,1,4)
        nfilters = 10 + int(90*x[0])                        #   in [10, 100]
        batch_size_train = int(pow(2.0, 4.0 + 4.0*x[1]))    #   in [2^4, 2^8] = [16, 256]
        M = float(x[2])                                     #   in [0, 1]
        LR = float(pow(10.0, -2 + 1.5*x[3]))                #   in [10^-2, 10^-0.5] = [0.01, ~0.31]

        return nfilters, batch_size_train, M, LR

    def write_new_line(self,filename,strng):
        f = open(filename,'w')
        f.write(strng)
        f.close()

    def append_new_line(self, filename, strng):
        f = open(filename, 'a')
        f.write(strng)
        f.close()

    def read_first_line(self,filename):
        f = open(filename,'r')
        line = f.readline()
        f.close()
        return line

    def read_last_line(self,filename):
        f = open(filename,'r')
        lines = f.readlines()
        if lines != []:
            lastline = lines[-1]
        else:
            lastline = ""
        f.close()
        return lastline

    def run_then_return_val_loss_parallel(self, hyperparameters, time_limit=100000, nepochs=100000):
        print("start parallel loop with nepochs: {}, time-limit: {}".format(nepochs, time_limit))
        stop_criterium = False
        val_loss_list = []
        # Counter keeps track of where we are in the hyperparameters list
        counter = 0
        counter_done = 0
        while(True):
            for filename in self.worker_file_list:
                time.sleep(1)
                lastline = self.read_last_line(filename).strip()
                print("Counters: {},{}".format(counter, counter_done))
                if counter_done == len(hyperparameters):
                    stop_criterium = True
                    break
                elif counter < len(hyperparameters) and lastline == "next":
                    # append configuration (=firstline) to solution file
                    line = self.read_first_line(filename)
                    self.append_new_line("solutions_ms", line)

                    # Structure of line is [nfilters, batch_size_train, M, LR, nepochs, val_loss, best_val_acc, running_time, nparameters]
                    val_loss_list.append(float(line.split()[6]))
                    print(val_loss_list)

                    # write new configuration to worker file
                    next_params = hyperparameters[counter]
                    nfilters = next_params[0]
                    batch_size_train = next_params[1]
                    M = next_params[2]
                    LR = next_params[3]
                    strng = "{}\t{}\t{}\t{}\t{}\t{}".format(nfilters,batch_size_train,M,LR,nepochs,time_limit)
                    self.write_new_line(filename,strng)

                    counter += 1
                    counter_done += 1
                elif counter >= len(hyperparameters) and lastline == "next":
                    # This worker is done and there are no new hyperparameter configurations
                    #   Read firstline from worker and tell him to wait
                    # append configuration (=firstline) to solution file
                    line = self.read_first_line(filename)
                    self.append_new_line("solutions_ms", line)

                    # Structure of line is [nfilters, batch_size_train, M, LR, nepochs, val_loss, best_val_acc, running_time, nparameters]
                    val_loss_list.append(float(line.split()[6]))
                    print(val_loss_list)

                    # Tell worker to wait
                    self.write_new_line(filename,"wait")

                    counter_done += 1
                elif counter >= len(hyperparameters) and lastline == "wait":
                    # This worker is waiting for work but there is no work and
                    #   other workers are still busy
                    pass
                elif counter < len(hyperparameters) and lastline == "wait":
                    # This worker is free for work and there is work
                    next_params = hyperparameters[counter]
                    nfilters = next_params[0]
                    batch_size_train = next_params[1]
                    M = next_params[2]
                    LR = next_params[3]
                    strng = "{}\t{}\t{}\t{}\t{}\t{}".format(nfilters,batch_size_train,M,LR,nepochs,time_limit)
                    self.write_new_line(filename,strng)

                    counter += 1

            if stop_criterium:
                break
        print("End parallel loop with nepochs: {}, time-limit: {}".format(nepochs, time_limit))
        return val_loss_list


    def hyperband(self):
        max_iter = 81   # maximum iterations/epochs per configuration
        eta = 3         # defines downsampling rate (default=3)
        logeta = lambda x: math.log(x)/math.log(eta)
        s_max = int(logeta(max_iter))   # number of unique executions of Successive Halving (minus one)
        B = (s_max+1)*max_iter          # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

        noiselevel = 0.2  # noise level of the objective function
        # Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.

        nruns = 1       # set it to e.g. 10 when testing hyperband against randomsearch
        for irun in range(0, 10):
            hband_results_filename = "stat_4/hyperband_{}.txt".format(irun)
            hband_file = open(hband_results_filename, 'w+', 0)

            x_best_observed = []
            x_best_observed_nep = 0

            nevals = 0       # total number of full (with max_iter epochs) evaluations used so far

            for s in reversed(range(s_max+1)):

                stat_filename = "stat_4/hband_benchmark_{}_{}.txt".format(irun,s)
                stat_file = open(stat_filename, 'w+', 0)

                n = int(math.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
                r = max_iter*eta**(-s)      # initial number of iterations to run configurations for

                # Begin Finite Horizon Successive Halving with (n,r)
                T = [ self.get_random_hyperparameter_configuration() for i in range(n) ]
                for i in range(s+1):
                    print("RUN: {}, {}, {}".format(irun, s, i))
                    # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                    n_i = n*eta**(-i)
                    r_i = r*eta**(i)
                    val_losses = self.run_then_return_val_loss_parallel(hyperparameters=T, time_limit=int(r_i/max_iter*60), nepochs=100000)

                    nevals = nevals + len(T) * r_i / 81
                    argsortidx = np.argsort(val_losses)

                    if (x_best_observed == []):
                        x_best_observed = T[argsortidx[0]]
                        y_best_observed = val_losses[argsortidx[0]]
                        x_best_observed_nep = r_i
                    # only if better AND based on >= number of epochs, the latter is optional
                    if (val_losses[argsortidx[0]] < y_best_observed):# and (r_i >= x_best_observed_nep):
                        x_best_observed_nep = r_i
                        y_best_observed = val_losses[argsortidx[0]]
                        x_best_observed = T[argsortidx[0]]

                    for j in range(0, len(T)):
                        stat_file.write("{}\t{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(
                                            T[j][0], T[j][1],T[j][2],T[j][3],r_i,val_losses[j]))
                    T = [ T[i] for i in argsortidx[0:int( n_i/eta )] ]

                    # suppose the current best solution w.r.t. validation loss is our recommendation
                    # then let's evaluate it in noiseless settings (~= averaging over tons of runs)
                    # if (len(T)):
                    #    f_recommendation = self.run_then_return_val_loss_parallel(81, [x_best_observed]) # 81 epochs and 1e-10 ~= zero noise
                hband_file.write("{}\t{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(
                                    x_best_observed[0], x_best_observed[1], x_best_observed[2], x_best_observed[3],
                                    x_best_observed_nep, y_best_observed))
                # End Finite Horizon Successive Halving with (n,r)

                stat_file.close()
            hband_file.close()

    def main(self, nb_workers=2):
        start = time.time()

        f = open("solutions_ms","w")
        f.close()

        for i in range(0,nb_workers):
            filename = "worker_{}".format(i)
            f = open(filename,'w')
            f.write("wait")
            f.close()
            self.worker_file_list.append(filename)

        self.hyperband()

        stop = time.time()
        total = stop - start
        f = open("timing","w")
        f.write("Total time: {}".format(total))
        f.close()

if __name__ == '__main__':
    m = Master()
    m.main(3)
