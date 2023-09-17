import subprocess
import math

add_counts = [ 10, 20, 40, 80, 160, 320, 640 ]
mul_counts = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 ]

for i in add_counts:
    for j in mul_counts:
        cmd = [ "./my_program.exe", str(i), str(j) ]
        output_str = subprocess \
            .run(cmd,stdout=subprocess.PIPE) \
            .stdout.decode('utf-8')
        time = float(output_str)
        print(f"{time},",end="")
    print()