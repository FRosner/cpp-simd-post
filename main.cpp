#include <iostream>
#include <vector>

void axpy(int n, float alpha, const std::vector<float>& x, std::vector<float>& y) {
    for (int i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main() {
    std::vector<float> x = {1.0, 2.0, 3.0};
    std::vector<float> y = {4.0, 5.0, 6.0};
    float alpha = 2.0;
    int n = x.size();

    axpy(n, alpha, x, y);

    for (float val : y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
