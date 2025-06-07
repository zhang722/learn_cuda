#include <iostream>

void upsweep(int *x, int n) {
    for (int s = 1; s < n; s *= 2) {
        for (int k = 0; k < n;) {
            int i = k + s - 1;
            int j = k + 2 * s - 1;
            x[j] += x[i];
            k = j + 1;
        } 
    }
}

void downsweep(int *x, int n) {
    x[n - 1] = 0;
    for (int s = n / 2; s > 0; s /= 2) {
        for (int k = 0; k < n;) {
            int i = k + s - 1;
            int j = k + 2 * s - 1;
            int temp = x[j];
            x[j] += x[i];
            x[i] = temp;
            k = j + 1; 
        } 
    }
}

void exclusive_scan(int *x, int n) {
    upsweep(x, n);
    downsweep(x, n);
}

int main() {
    int a[] = {1, 1, 1, 1};
    exclusive_scan(a, 4);
    for (int i = 0; i < 4; i++) {
        std::cout << a[i] << ','; 
    }

    std::cout << std::endl;
}