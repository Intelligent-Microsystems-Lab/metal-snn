import os

ident_word = "UnQuantized"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q " 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython snn_dnmnist.py"


sweep_parameters = {'nclasses':[4,5,6]}

trials = 3

avail_q = ['gpu@qa-rtx6k-040.crc.nd.edu', 'gpu@qa-rtx6k-041.crc.nd.edu']
#avail_q = ['gpu@@joshi']
q_counter = 0

for i in range(trials):
    for variable in sweep_parameters:
        for value in sweep_parameters[variable]:
            name = ident_word + "_" +variable + "_" + str(value) + "_" + str(i)
            with open('jobscripts/'+name+'.script', 'w') as f:
                f.write(part1 + avail_q[q_counter] + part11  + name + part2 + name + part3 + name + part4 + " --" + variable + " " + str(value))
            os.system("qsub "+ 'jobscripts/'+name+'.script')
            q_counter += 1
            if q_counter >= len(avail_q):
                q_counter = 0