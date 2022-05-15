#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// gcc -std=c99 seqfactorize.c -lm
// -lm : https://stackoverflow.com/questions/10409032/why-am-i-getting-undefined-reference-to-sqrt-error-even-though-i-include-math

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Wrong number of arguments. Exit!\n");
	 	exit(1);
    }

    unsigned int N = atoi(argv[1]);

    // While N is divisible by 2, print 2 and divide N by 2
    while (N % 2 == 0) {
    	printf("2 ");
    	N /= 2;
    }

    // After step 1, N must be odd
    // Now start a loop from i = 3 to ceiling(square root of N)
    for (unsigned int i = 3; i < ceil(sqrt(N)); i += 2) { // increment i by 2
        // print i
    	while (N % i == 0) {
            // print i
    		printf("%u ", i);
            // keep dividing N by i and print i till N is no longer divisible by i
    		N /= i;
    	}
        // continue the outer loop
    }

    // If the remaining of N is a prime number and is greater than 2, then N will not become 1 by above two steps. So print N if it is greater than 2
    if (N > 2) {
    	printf("%u\n", N);
        return 0;
    }

    printf("\n");
    return 0;
}