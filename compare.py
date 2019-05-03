import numpy as np 
from shutil import copyfile

f1 = open ("log.txt", "r")
f2 = open ("log_last.txt", "r")



mp_last = {}
lines = f2.readlines ()
for i in range (len (lines)):
	l = lines[i].split ()
	l[0] = float (l[0])
	l[1] = int (l[1])
	mp_last [(l[0], l[1])] = [float (v) for v in l [2:]]

mp_current = {}
lines = f1.readlines ()
for i in range (len (lines)):
	l = lines[i].split ()
	l[0] = float (l[0])
	l[1] = int (l[1])
	mp_current [(l[0], l[1])] = [float (v) for v in l [2:]]
	if (mp_last [(l[0], l[1])] != mp_current [(l[0], l[1])]):
		print ("DIFF")
		print ("CURRENT: ", l [0:2] + mp_current [(l[0], l[1])])
		print ("LAST   : ", l [0:2] + mp_last [(l[0], l[1])])
		diff = []
		for k in range (len (mp_current [(l[0], l[1])])):
			diff += [abs (mp_current [(l[0], l[1])] [k] - mp_last [(l[0], l[1])][k])]
		print ("DIFFS", diff)
		print ("MAX_DIF", np.max (diff))

		print ()
		# exit ()

print ("SAME")
# copyfile ("log.txt", "log_last.txt")

