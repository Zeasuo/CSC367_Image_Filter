import matplotlib
matplotlib.use('Agg')
import subprocess
import re
import numpy as np
from matplotlib import pyplot as plt

ITER = 10

def test_size(size):
    cpu, k1, k2, k3, k4, k5 = [], [], [], [], [], []
    for i in range(ITER):
        proc = subprocess.Popen(["./main", "-i", size+".pgm", "-o", size+"result.pgm"], stdout=subprocess.PIPE)
        proc.stdout.readline()
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        cpu.append([float(output_split[1]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k1.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k2.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k3.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k4.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k5.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
    kernel_names = ["CPU", "K1", "K2", "K3", "K4", "K5"]
    kernels = [cpu, k1, k2, k3, k4, k5]
    sums = []
    pro_times = []

    for i in range(6):
        sum = 0
        pro_time = 0
        for run in kernels[i]:
            pro_time += run[0]
            sum += np.sum(run)
        sums.append(sum/ITER)
        pro_times.append(pro_time/ITER)
        
    plt.xlabel("Process Method")
    plt.ylabel("Time (ms)")
    plt.title("Total Time For CPU and Kernels. Average over 10 runs ("+size+")")
    plt.bar(kernel_names, sums)
    plt.savefig('Total time'+size+'.png',dpi=800)
    plt.clf()
    plt.title("Execution Time For CPU and Kernels. Average over 10 runs ("+size+")")
    plt.xlabel("Process Method")
    plt.ylabel("Time (ms)")
    plt.bar(kernel_names, pro_times)
    plt.savefig('Execution time'+size+'.png',dpi=800)
    plt.clf()


if __name__ == "__main__":
    test_size("1mb")
    test_size("10mb")
    test_size("50mb")
    test_size("100mb")