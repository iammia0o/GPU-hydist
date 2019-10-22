running instruction: 

- files involved:
	+ rep.py 
	+ constant.cuh
	+ Global_Variables.py
	+ .cu
- pre-run: for each case (song luy, tan chau, synthesis), copy coresponding contanst file to the constant.cuh. 
		ex: running for Tan Chau, replace constant_Tanchau.cuh to constant.cuh 
- running : py rep.py 
	     parameters:
		"--Device" : device to launch the program
		"--mins", : total running minute
		"--hours" : total running hour
		"--plot", default=False : plot u, v, z
		"--test",default=False : parameter for debugging
		"--sediment" : hour at which we start running sediment module
		"--bed_change" : hour at which we start run bed change module
		"--pick_up" : load intial condition or not
		"--pickup_dirs" : directory to load initial condition
		"--kenhhepng" : this mean code for synthesis data would be used, this is horizontal channel
		"--kenhhepd"  : code for synthesis data would be used, this is vertical channel
