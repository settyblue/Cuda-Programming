// initialize an array using GPU
#include<iostream>
using namespace std;
#define DEBUG 0
#define BLOCK_SIZE_X 17
#define BLOCK_SIZE_Y 17
#define GRID_SIZE_X 241
#define GRID_SIZE_Y 241
#define BLOCK_SIZE BLOCK_SIZE_X*BLOCK_SIZE_Y
#define GRID_SIZE GRID_SIZE_X*GRID_SIZE_Y
#define NUMBER_OF_TILES_X 1
#define NUMBER_OF_TILES_Y 1
#define NUMBER_OF_TILES NUMBER_OF_TILES_X*NUMBER_OF_TILES_Y
#define ARRAY_SIZE BLOCK_SIZE*GRID_SIZE*NUMBER_OF_TILES
#define ARRAY_SIZE_X BLOCK_SIZE_X*GRID_SIZE_X*NUMBER_OF_TILES_X
#define ARRAY_SIZE_Y BLOCK_SIZE_Y*GRID_SIZE_Y*NUMBER_OF_TILES_Y

// kernel.
__global__ void initArray( double *A, double *B){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ROW, COL;
	for(int k=0;k<100;k++){
		for(int m=0;m<NUMBER_OF_TILES_X;m++){
			for(int n=0;n<NUMBER_OF_TILES_Y;n++){
				ROW = row+m*gridDim.x*blockDim.x;
				COL = col+n*gridDim.y*blockDim.y;
				if(ROW < 4097 && COL < 4097){
					if(ROW == 0 || COL == gridDim.y*blockDim.y*NUMBER_OF_TILES_Y-1){
						//Do Nothing.
					}else{
						B[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)] 
							= A[((ROW)-1)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)+1]+A[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)+1];
					}
					__syncthreads();
					A[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y + COL] = B[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y + (COL)];
					__syncthreads();
				}
			}
		}	
	}//end of k loop.
}

int main(){
	double *h_A; //pointer for host memory
	double *d_A,*d_B; //pointer for device memory
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// define thread hierarchy
	// allocate host and device memory
	size_t memSize;
	memSize = BLOCK_SIZE * GRID_SIZE * NUMBER_OF_TILES * sizeof(double);
	h_A = (double*) malloc(memSize);
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &d_B, memSize);
	//cout<<"debug 1."<<endl;
	//Initialize host array values.
	for(int i=0;i<ARRAY_SIZE_X;i++){
		for(int j=0;j<ARRAY_SIZE_Y;j++){
			if(i<4097 && j <4097){
				h_A[i*ARRAY_SIZE_Y+j] = 1 + (rand()%11)*0.1;
			}
			else{
				h_A[i*ARRAY_SIZE_Y+j] = 0;
			}
		}
	}
	//print the host array
	if(DEBUG==1){
		for (int i=0;i<ARRAY_SIZE_X;i++){
			for (int j=0;j<ARRAY_SIZE_Y;j++){
				cout<<h_A[i*ARRAY_SIZE_Y+j]<<"\t";
			}
			cout<<endl;
		}
	}
	//copy the array to device.
	cudaMemcpy(d_A, h_A, memSize,cudaMemcpyHostToDevice);
	// launch kernel
	dim3 dimGrid(GRID_SIZE_X,GRID_SIZE_Y);
	dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	
	cudaEventRecord(start);
	initArray<<< dimGrid, dimBlock >>>(d_A,d_B);
	cudaEventRecord(stop);
	
	// get results
	cudaMemcpy(h_A, d_A, memSize,cudaMemcpyDeviceToHost);
	//verify
	//print the host array
	if(DEBUG==1){
		for (int i=0;i<ARRAY_SIZE_X;i++){
			for (int j=0;j<ARRAY_SIZE_Y;j++){
				cout<<h_A[i*ARRAY_SIZE_Y+j]<<"\t";
			}
			cout<<endl;
		}
	}
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"Time taken to complete : "<<milliseconds<<" milliseconds"<<endl;
	cout<<endl;
}
