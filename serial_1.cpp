#include<iostream>
#include<time.h>
using namespace std;

int main(){
	clock_t t;
	double F[4097][4097];
	t = clock();
	// insert code to initialize array elements to random values between 1.0 and 2.0 
	for (int k = 0; k < 100; k++) 
		for (int i = 1; i < 4097; i++) 
			for (int j=0; j < 4096; j++) 
				F[ i ][ j ] = F[ i-1 ][ j+1 ] + F[ i ][ j+1 ];

	t = clock() - t;
	cout<<"Time taken to complete : "<<(((float)t)/CLOCKS_PER_SEC)<<" seconds"<<endl;
}
