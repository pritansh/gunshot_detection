import numpy as np
from datetime import datetime

final = np.load('./kartik/d-1-final.npy')
faults = np.load('./kartik/d-1-faults.npy')

f_d = np.empty(0, dtype=int)
f_f = np.empty(0, dtype=int)
f_a = []
i, j, k = (0, 0, 1)
str_format = '%Y-%m-%d'

while i < len(final):
    f_d = np.hstack([f_d, k])
    f_f = np.hstack([f_f, faults[i]])
    f_a.append([final[i].strftime(str_format)])
    j = i + 1
    prev = final[i]
    while j < len(final):
        temp = final[j] - prev
        if temp.days <= 7:
            f_f[k-1] += faults[j]
            f_a[k-1].append(final[j].strftime(str_format))
            i += 1
            j += 1
        else:
            j = len(final)
    i += 1
    k += 1

for i in range(0, len(f_a)):
    print f_a[i], i+1