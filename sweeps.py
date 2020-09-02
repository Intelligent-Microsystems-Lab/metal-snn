import os

ident_word = "ASL-DVS"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q " 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython test.py"


#sweep_parameters = {'n-train':[15,25],'train-samples':[200]}
#sweep_parameters = {'n-train':[10,15,25], 'train-samples':[200]}
#sweep_parameters = {'checkpoint':['039b8454-1d57-4534-801a-6ae170dc368c','206b224c-20c8-41a3-884b-400d8a6b6677','2705f28d-5cf7-4b4e-8c4a-d1eeeee7108f','4dc5d819-c5a8-445e-aac3-db6cf90e484c','9b3bc513-e21f-4399-bae9-8220b32e2884','bd1ab2b1-7d98-45a2-8cb7-70fcccbccb18','f55bbde6-98ff-4fa0-be2b-ddaf6de7ac1c','f80d7f4a-0a32-4b5a-a2fe-380d9e9c0dae']}

sweep_parameters = {'batch-size':[32]}
trials = 4

#avail_q = ['gpu@qa-rtx6k-040.crc.nd.edu', 'gpu@qa-rtx6k-041.crc.nd.edu']
avail_q = ['gpu@@joshi']
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




