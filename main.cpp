#if USE_GPU 
#include "cpu_matrix.h"
#include "gpu_matrix.h"
typedef gpu_matrix matrix;
#else
#include "cpu_matrix.h"
typedef cpu_matrix matrix;
#endif

int main(){
	InitGPU(DEVICE::getInstance(), 1000000000, 0);
	matrix a;
	a.init(3, 4);	
	
	return 0;
}
