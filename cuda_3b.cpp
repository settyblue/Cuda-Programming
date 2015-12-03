// initialize an array using GPU
//cuda program to reverse contents of an array.
//creating one block of 1024 threads each.
#include<iostream>
#include<time.h>
using namespace std;
#define DEBUG 0
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define GRID_SIZE_X 1
#define GRID_SIZE_Y 1
#define BLOCK_SIZE BLOCK_SIZE_X*BLOCK_SIZE_Y
#define GRID_SIZE GRID_SIZE_X*GRID_SIZE_Y
#define NUMBER_OF_TILES_X 32
#define NUMBER_OF_TILES_Y 32
#define NUMBER_OF_TILES NUMBER_OF_TILES_X*NUMBER_OF_TILES_Y
#define ARRAY_SIZE BLOCK_SIZE*GRID_SIZE*NUMBER_OF_TILES
#define ARRAY_SIZE_X BLOCK_SIZE_X*GRID_SIZE_X*NUMBER_OF_TILES_X
#define ARRAY_SIZE_Y BLOCK_SIZE_Y*GRID_SIZE_Y*NUMBER_OF_TILES_Y

// kernel.
__global__ void initArray( double *A){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ROW, COL;
	int temp;
	for(int m=0;m<NUMBER_OF_TILES_X;m++){
		for(int n=0;n<NUMBER_OF_TILES_Y;n++){
			ROW = row+m*gridDim.x*blockDim.x;
			COL = col+n*gridDim.y*blockDim.y;
			if(ROW < ARRAY_SIZE_X/2){
				temp = A[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)] ;
				A[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)] = 
					A[ARRAY_SIZE - ((ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)) - 1];
				A[ARRAY_SIZE - ((ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)) - 1] = temp;
			}else{
				//skip.
			}
		}
	}
}

int main(){

	double *h_A; //pointer for host memory
	double *h_B;
	double *d_A; //pointer for device memory
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// define thread hierarchy
	// allocate host and device memory
	size_t memSize;
	memSize = BLOCK_SIZE * GRID_SIZE * NUMBER_OF_TILES * sizeof(double);
	h_A = (double*) malloc(memSize);
	h_B = (double*) malloc(memSize);
	cudaMalloc( (void**) &d_A, memSize);
	//Initialize host array values.
	for(int i=0;i<ARRAY_SIZE_X;i++){
		for(int j=0;j<ARRAY_SIZE_Y;j++){
			h_A[i*ARRAY_SIZE_Y+j] =  1.0 + (rand()%1000);
			h_B[i*ARRAY_SIZE_Y+j] = h_A[i*ARRAY_SIZE_Y+j];
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
	cout<<"dimensions of the array : "<<ARRAY_SIZE_X<<" * "<<ARRAY_SIZE_Y<<endl;
	cout<<"creating kernel with one block of 1024 threads each."<<endl;
	cout<<"dimensions of the grid : "<<GRID_SIZE_X<<" * "<<GRID_SIZE_Y<<endl;
	cout<<"dimensions of a block : "<<BLOCK_SIZE_X<<" * "<<BLOCK_SIZE_Y<<endl;
	cout<<"launching kernel to reverse the contents of the array."<<endl;
	dim3 dimGrid(GRID_SIZE_X,GRID_SIZE_Y);
	dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	cudaEventRecord(start);
	initArray<<< dimGrid, dimBlock >>>(d_A);
	cudaEventRecord(stop);
	// get results
	cudaMemcpy(h_A, d_A, memSize,cudaMemcpyDeviceToHost);
	//print the host array
	if(DEBUG==1){
		for (int i=0;i<ARRAY_SIZE_X;i++){
			for (int j=0;j<ARRAY_SIZE_Y;j++){
				cout<<h_A[i*ARRAY_SIZE_Y+j]<<"\t";
			}
			cout<<endl;
		}
	}
	cout<<endl;
	//verification.
	bool verified = true;
	for (int i=0;i<ARRAY_SIZE_X;i++){
		for (int j=0;j<ARRAY_SIZE_Y;j++){
			if( h_A[i*ARRAY_SIZE_Y+j] != h_B[ARRAY_SIZE_X*ARRAY_SIZE_Y - (i*ARRAY_SIZE_Y+j) -1]){
				verified = false;
				break;
			}
		}
	}
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(verified){
		cout<<"Reversing the Matrix was successful."<<endl;
	}else{
		cout<<"matrix was incorrectly reversed."<<endl;
	}
	cout<<"Time taken to complete parallel version : "<<milliseconds<<" milliseconds"<<endl;
	cout<<endl;
}
