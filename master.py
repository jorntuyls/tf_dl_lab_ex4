
import numpy as np
import time
import math

from worker import Worker

class Master:

    def __init__(self):
        self.worker = Worker()

    def get_random_hyperparameter_configuration(self):
        x = np.random.uniform(0,1,4)
        nfilters = 10 + int(90*x[0])                        #   in [10, 100]
        batch_size_train = int(pow(2.0, 4.0 + 4.0*x[1]))    #   in [2^4, 2^8] = [16, 256]
        M = float(x[2])                                     #   in [0, 1]
        LR = float(pow(10.0, -2 + 1.5*x[3]))                #   in [10^-2, 10^-0.5] = [0.01, ~0.31]

        return nfilters, batch_size_train, M, LR

    def open_file(self,filename):
        f = open(filename,'w')
        f.close()

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

    def run_then_return_val_loss(self, hyperparameters, time_limit=100000, nepochs=100000):
        ntrain = 510          # the whole training set
        nvalid = 510         #
        ntest = 510           #
        batch_size_valid = 500  # does not influence training process, but reduces time loss from validation
        batch_size_test = 500   # same here

        algorithm_type = 2  # SGD with momentum
        irun = 1  # one run only
        mexevaluations = 200
        nvariables = 4

        val_loss_list = []

        for param in hyperparameters:
            nfilters = param[0]
            batch_size_train = param[1]
            M = param[2]
            LR = param[3]
            result = self.worker.get_result(ntrain, nvalid, ntest, algorithm_type, batch_size_train, batch_size_valid, batch_size_test, nepochs, "stat.txt", LR, M, nfilters, time_limit)
            val_loss_list.append(result[0])

        return val_loss_list
