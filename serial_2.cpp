#include<iostream>
#include<stdlib.h>
#include<time.h>
#define size 4096
using namespace std;

int main(){
	clock_t t;
	double A[size][size], B[size][size];
	t = clock();
	for (int i = 0; i < size; i++){
		for(int j = 0;j < size; j++){
			A[ i ][ j ] = 1 + (rand()%2)*0.1;
		}
	} 
	for (int i = 0; i < size; i++) 
		for (int j = 0; j < size; j++) 
			for (int k = 0; k < size; k++) 
				B[ i ][ j ] = A[ i ][ k ]*A[ j ][ k ];

	t = clock() - t;
	cout<<"Time taken to complete : "<<(((float)t)/CLOCKS_PER_SEC)<<" seconds"<<endl;
}