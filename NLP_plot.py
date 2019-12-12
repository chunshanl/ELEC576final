import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np

# -----------
# plot test error
# ------------
# with open('Q2_accuracy_and_loss_LSTM.txt',
# newline='') as csvfile:

#     data = list(csv.reader(csvfile))
################
with open('cifar_seed0.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))

with open('cifar_seed1.txt', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))

with open('cifar_seed2.txt', newline='') as csvfile:
    data3 = list(csv.reader(csvfile))

with open('cifar_seed3.txt', newline='') as csvfile:
    data4 = list(csv.reader(csvfile))

#############################
with open('cifar_add_seed0.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))

with open('cifar_add_seed1.txt', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))

with open('cifar_add_seed2.txt', newline='') as csvfile:
    data3 = list(csv.reader(csvfile))

with open('cifar_add_seed3.txt', newline='') as csvfile:
    data4 = list(csv.reader(csvfile))

###############################
with open('cifar_mul_seed0.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))

with open('cifar_mul_seed1.txt', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))

with open('cifar_mul_seed2.txt', newline='') as csvfile:
    data3 = list(csv.reader(csvfile))

with open('cifar_mul_seed3.txt', newline='') as csvfile:
    data4 = list(csv.reader(csvfile))

################
with open('NLP_6_seed4.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))

with open('NLP_6_seed5.txt', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))

with open('NLP_6_seed6.txt', newline='') as csvfile:
    data3 = list(csv.reader(csvfile))

with open('NLP_6_seed7.txt', newline='') as csvfile:
    data4 = list(csv.reader(csvfile))

##############3
with open('NLP_add_seed0.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))

with open('NLP_add_seed1.txt', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))

with open('NLP_add_seed2.txt', newline='') as csvfile:
    data3 = list(csv.reader(csvfile))

with open('NLP_add_seed3.txt', newline='') as csvfile:
    data4 = list(csv.reader(csvfile))
################3
with open('NLP_mul_seed0.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))

with open('NLP_mul_seed1.txt', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))

with open('NLP_mul_seed2.txt', newline='') as csvfile:
    data3 = list(csv.reader(csvfile))

with open('NLP_mul_seed3.txt', newline='') as csvfile:
    data4 = list(csv.reader(csvfile))


for ii in range(6):
    a1 = data1[ii]
    a1 = [float(i) for i in a1]
    plt.plot(a1, marker='', color='grey', linewidth=1, alpha=0.4)
    a2 = data2[ii]
    a2 = [float(i) for i in a2]
    plt.plot(a2, marker='', color='grey', linewidth=1, alpha=0.4)
    a3 = data3[ii]
    a3 = [float(i) for i in a3]
    plt.plot(a3, marker='', color='grey', linewidth=1, alpha=0.4)
    a4 = data4[ii]
    a4 = [float(i) for i in a4]
    plt.plot(a4, marker='', color='grey', linewidth=1, alpha=0.4)
    b=np.mean(np.array([a1, a2, a3, a4]), axis=0)
    plt.plot(b, marker='', color='dodgerblue', linewidth=2)
    # plt.ylim([0,1])
    plt.show()

print(b[-1])
print(a1[-1])
print(a2[-1])
print(a3[-1])
print(a4[-1])


with open('cifar_seed0.txt', newline='') as csvfile:
    data1 = list(csv.reader(csvfile))


