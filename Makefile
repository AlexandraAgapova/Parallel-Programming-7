cuda = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib -lcublas -lcudart
gpu:
	pgc++ -o main -lboost_program_options ${cuda} -acc=gpu -Minfo=all  main.cpp
	./main --size 512 --tol 1e-6 --max_iter 1000000

profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task --size=512 --max_iter=30
