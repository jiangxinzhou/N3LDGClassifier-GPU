#ifndef N3LDG_ALL
#define N3LDG_ALL


#if USE_GPU
	#include "cpu_matrix.h"
	#include "gpu_matrix.h"
	typedef gpu_matrix matrix;
#else
	#include "cpu_matrix.h"
	typedef  cpu_matrix matrix;
#endif

#include "Graph.h"
#include "Node.h"
#include "Alphabet.h"
#include "NRMat.h"
#include "MyLib.h"
#include "Metric.h"



#include "BucketOP.h"
#include "LookupTable.h"
#include "Param.h"
#include "SparseParam.h"

#include "ModelUpdate.h"
#include "CheckGrad.h"
#include "Pooling.h"
#include "Concat.h"
#include "Windowlized.h"
#include "UniOP.h"




#include "SoftMaxLoss.h"







#endif
