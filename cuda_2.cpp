// cuda program to multiply the array with the transpose of itself.
#include<iostream>
#include<time.h>
using namespace std;
#define DEBUG 0
#define SUPPRESS_SERIAL 1
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define GRID_SIZE_X 256
#define GRID_SIZE_Y 256
#define BLOCK_SIZE BLOCK_SIZE_X*BLOCK_SIZE_Y
#define GRID_SIZE GRID_SIZE_X*GRID_SIZE_Y
#define NUMBER_OF_TILES_X 1
#define NUMBER_OF_TILES_Y 1
#define NUMBER_OF_TILES NUMBER_OF_TILES_X*NUMBER_OF_TILES_Y
#define ARRAY_SIZE BLOCK_SIZE*GRID_SIZE*NUMBER_OF_TILES
#define ARRAY_SIZE_X BLOCK_SIZE_X*GRID_SIZE_X*NUMBER_OF_TILES_X
#define ARRAY_SIZE_Y BLOCK_SIZE_Y*GRID_SIZE_Y*NUMBER_OF_TILES_Y

// kernel.
__global__ void initArray( double *A, double *C){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ROW, COL;
	for(int m=0;m<NUMBER_OF_TILES_X;m++){
		for(int n=0;n<NUMBER_OF_TILES_Y;n++){
			ROW = row+m*gridDim.x*blockDim.x;
			COL = col+n*gridDim.y*blockDim.y;
			for(int i=0;i<ARRAY_SIZE_X;i++){
				C[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+(COL)] += 
					A[(ROW)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+i]
					*A[(COL)*blockDim.y*gridDim.y*NUMBER_OF_TILES_Y+i];
			}
		}
	}
}

// host function.
int main(){
	clock_t t;
	double *h_A; //pointer for host memory
	double *h_B; //pointer for host array
	double *h_C; //pointer for host memory
	double *d_A; //pointer for device memory
	double *d_C; //pointer for device memory
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// define thread hierarchy
	// allocate host and device memory
	size_t memSize;
	memSize = BLOCK_SIZE * GRID_SIZE * NUMBER_OF_TILES * sizeof(double);
	h_C = (double*) malloc(memSize);
	h_B = (double*) malloc(memSize);
	h_A = (double*) malloc(memSize);
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &d_C, memSize);
	//Initialize host array values.
	for(int i=0;i<ARRAY_SIZE_X;i++){
		for(int j=0;j<ARRAY_SIZE_Y;j++){
			h_A[i*ARRAY_SIZE_Y+j] =  1.0 + (rand()%11)*0.1;
			h_C[i*ARRAY_SIZE_Y+j] =  0.0 ;
			h_B[i*ARRAY_SIZE_Y+j] =  0.0 ;
		}
	}
	
	if(SUPPRESS_SERIAL == 0){
		//running the serial version.
		t = clock();
		for (int i = 0; i < ARRAY_SIZE_X; i++) 
			for (int j = 0; j < ARRAY_SIZE_Y; j++) 
				for (int k = 0; k < ARRAY_SIZE_Y; k++) 
					h_B[i*ARRAY_SIZE_Y + j] += h_A[i*ARRAY_SIZE_Y + k] * h_A[j*ARRAY_SIZE_Y + k];
		t = clock() - t;
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
	cudaMemcpy(d_C, h_C, memSize,cudaMemcpyHostToDevice);
	// launch kernel
	cout<<"dimensions of the array : "<<ARRAY_SIZE_X<<" * "<<ARRAY_SIZE_Y<<endl;
	cout<<"launching kernel to multiply the array with the transpose of itself."<<endl;
	dim3 dimGrid(GRID_SIZE_X,GRID_SIZE_Y);
	dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);

	cudaEventRecord(start);
	initArray<<< dimGrid, dimBlock >>>(d_A,d_C);
	cudaEventRecord(stop);
	// get results
	cudaMemcpy(h_C, d_C, memSize,cudaMemcpyDeviceToHost);
	//verify
	//print the host array
	//int count = 0;
	if(DEBUG==1){
		cout<<endl;
		cout<<"printing parallel version result"<<endl;
		for (int i=0;i<ARRAY_SIZE_X;i++){
			for (int j=0;j<ARRAY_SIZE_Y;j++){
				cout<<h_C[i*ARRAY_SIZE_Y+j]<<"\t";
			}
			cout<<endl;
		}
	}

	if(DEBUG==1){
		cout<<endl;
		cout<<"printing serial version result"<<endl;
		for (int i=0;i<ARRAY_SIZE_X;i++){
			for (int j=0;j<ARRAY_SIZE_Y;j++){
				cout<<h_B[i*ARRAY_SIZE_Y+j]<<"\t";
			}
			cout<<endl;
		}
	}
	//verify serial and paralle results.
	bool equal = true;
	if(SUPPRESS_SERIAL == 0){
		for(int i=0;i<ARRAY_SIZE_X;i++){
			for(int j=0;j<ARRAY_SIZE_Y;j++){
				if(h_B[i*ARRAY_SIZE_Y+j] != h_C[i*ARRAY_SIZE_Y+j]){
					equal = false;
					break;
				}
			}
		}	
	}
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(SUPPRESS_SERIAL == 0){
		if(equal)
			cout<<"serial and parallel version results are same."<<endl;
		else
			cout<<"serial and parallel version results are not same."<<endl ;
		cout<<"Time taken to complete serial version : "<<(((float)t)/CLOCKS_PER_SEC)<<" seconds"<<endl;
	}
	cout<<"Dimensions of the grid : "<<GRID_SIZE_X<<" * "<<GRID_SIZE_Y<<endl;
	cout<<"Dimensions of a block : "<<BLOCK_SIZE_X<<" * "<<BLOCK_SIZE_Y<<endl;
	cout<<"Time taken to complete parallel version : "<<milliseconds<<" milliseconds"<<endl;
	cout<<endl;
}
