#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "dynamic/nn.hpp"
#include "dynamic/optim.hpp"
#include "common.hpp"
#include <iostream>
#include <iomanip>

using namespace autograd;

// MSE Loss: mean((pred - target)^2)
inline ValuePtr mse_loss(ValuePtr pred, ValuePtr target) {
    auto diff = sub(pred, target);
    auto sq = mul(diff, diff);
    return reduce_mean(sq);
}

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Simple test: Linear -> MSE loss
    Linear linear(32, 32, device);
    SGD sgd(linear.parameters(), 0.01f);
    
    auto x = make_ones(ttnn::Shape({32, 32}), device, true, "x");
    auto target = make_zeros(ttnn::Shape({32, 32}), device, false, "target");
    
    // Get initial weight
    auto w_before = linear.weight->data.cpu().to_vector<bfloat16>()[0];
    
    // Forward
    sgd.zero_grad();
    auto pred = linear.forward(x);
    auto loss = mse_loss(pred, target);
    
    // Backward
    backward(loss, device);
    
    // Check if weight has gradient
    if (linear.weight->grad.has_value()) {
        auto grad = linear.weight->grad.value().cpu().to_vector<bfloat16>()[0];
        std::cout << "weight grad[0]: " << static_cast<float>(grad) << std::endl;
    } else {
        std::cout << "ERROR: weight has no gradient!" << std::endl;
    }
    
    // Step
    sgd.step();
    
    // Check weight changed
    auto w_after = linear.weight->data.cpu().to_vector<bfloat16>()[0];
    std::cout << "weight[0] before: " << static_cast<float>(w_before) << std::endl;
    std::cout << "weight[0] after:  " << static_cast<float>(w_after) << std::endl;
    std::cout << "delta: " << (static_cast<float>(w_after) - static_cast<float>(w_before)) << std::endl;
    
    return 0;
}
