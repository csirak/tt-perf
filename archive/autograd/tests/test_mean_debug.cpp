#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "common.hpp"
#include <iostream>
#include <iomanip>

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Create a small test case
    auto x = make_ones(ttnn::Shape({4, 4}), device, true, "x");
    
    // Test reduce_mean
    auto y = reduce_mean(x);
    
    std::cout << "x shape: " << x->shape() << std::endl;
    std::cout << "y shape: " << y->shape() << std::endl;
    
    // y should be 1.0 (mean of all ones)
    auto y_val = y->data.cpu().to_vector<bfloat16>()[0];
    std::cout << "y value: " << static_cast<float>(y_val) << " (expected 1.0)" << std::endl;
    
    // Backward
    backward(y, device);
    
    // Check gradient: should be 1.0 / 16 = 0.0625 everywhere
    auto x_grad = x->grad.value().cpu().to_vector<bfloat16>();
    std::cout << "x grad[0]: " << static_cast<float>(x_grad[0]) << " (expected 0.0625)" << std::endl;
    
    // Test with larger tensor
    auto x2 = make_ones(ttnn::Shape({1024, 128}), device, true, "x2");
    auto y2 = reduce_mean(x2);
    backward(y2, device);
    
    auto x2_grad = x2->grad.value().cpu().to_vector<bfloat16>();
    float expected_grad = 1.0f / (1024 * 128);
    std::cout << "x2 grad[0]: " << std::scientific << static_cast<float>(x2_grad[0]) 
              << " (expected " << expected_grad << ")" << std::endl;
    
    return 0;
}
