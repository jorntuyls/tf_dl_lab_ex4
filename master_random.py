
import numpy as np
import time
import math

from master import Master
from worker import Worker

class RandomMaster(Master):

    def __init__(self):
        self.worker_file_list = []
        self.solution_file = "stat_random/solution_ms_random"
        self.open_file(self.solution_file)
        self.worker = Worker()

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
                    self.append_new_line(self.solution_file, line)

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
                    self.append_new_line(self.solution_file, line)

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

    def random_search(self, num_confs=100, time_limit=100000, nepochs=100000, parallel=True):

        for irun in range(8,10):
            start_time = time.time()
            stat_file = open("stat_random/random_{}.txt".format(irun),'w+',0)
            x_best_observed = []
            y_best_observed = 0

            for i in range(0,num_confs):
                conf = self.get_random_hyperparameter_configuration()
                if parallel:
                    val_loss_list = self.run_then_return_val_loss_parallel([conf], time_limit=time_limit, nepochs=nepochs)
                else:
                    val_loss_list = self.run_then_return_val_loss([conf], time_limit=time_limit, nepochs=nepochs)

                if (x_best_observed == []) or (val_loss_list[0] < y_best_observed):
                    x_best_observed = conf
                    y_best_observed = val_loss_list[0]

                stat_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1, time.time() - start_time, x_best_observed[0],
                            x_best_observed[1], x_best_observed[2], x_best_observed[3], y_best_observed))

        stat_file.close()


    def main(self, nb_workers=2):

        for i in range(0,nb_workers):
            filename = "worker_{}".format(i)
            f = open(filename,'w')
            f.write("wait")
            f.close()
            self.worker_file_list.append(filename)

        self.random_search(num_confs=15, time_limit=60, parallel=False)

if __name__ == '__main__':
    m = RandomMaster()
    m.main(0)
