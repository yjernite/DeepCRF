#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("ChainCRF")
    .Input("log_potentials: float32")
    .Input("tags: int32")
    .Input("forward_sp: float32")
    .Input("backward_sp: float32")
    .Input("gradients: float32")
    .Output("log_likelihood: float32")
    .Output("marginals: float32")
    .Doc(R"doc(
    A module which performs inference on a chain CRF to obtain the
    partition function and single-node marginals.
    )doc");

using namespace tensorflow;

float LogSumExpP(float nums[], size_t ct) {
	float max_exp = nums[0], sum = 0.0;
	size_t i;
	for (i = 1 ; i < ct ; i++)
		if (nums[i] > max_exp)
			max_exp = nums[i];
	for (i = 0; i < ct ; i++)
		sum += exp(nums[i] - max_exp);
	return log(sum) + max_exp;
}

class ChainCRFOp : public OpKernel {
    public: explicit ChainCRFOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& log_potentials = context->input(0);
        auto log_pots = log_potentials.tensor<float, 3>(); // TODO: replace 3
        
        const Tensor& in_tags = context->input(1);
        auto tags = in_tags.tensor<int, 1>();
        
        const Tensor& in_forward_sp = context->input(2);
        auto forward_sp = in_forward_sp.tensor<float, 2>();
        
        const Tensor& in_backward_sp = context->input(3);
        auto backward_sp = in_backward_sp.tensor<float, 2>();
        
        const Tensor& gradients = context->input(4);
        
        int seq_length = log_potentials.dim_size(0);
        int n_vars = log_potentials.dim_size(1);
        
        float aux_array[n_vars];
        
        // prepare TensorShape vectors
        TensorShape marg_shape;
        marg_shape.AddDim(seq_length);
        marg_shape.AddDim(n_vars);
        
        // Create an output tensor: partition
        Tensor* out_ll = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, {1},
                                                         &out_ll));
        
        // Compute partition function
        auto ll = out_ll->tensor<float, 1>();
        for (int j =0; j < n_vars; j++)
            aux_array[j] = backward_sp(0, j);
        ll(0) = -LogSumExpP(aux_array, n_vars);
        for (int i = 1; i < seq_length; i++)
			ll(0) += log_pots(i, tags(i-1), tags(i));
        
        // Create an output tensor: marginals
        Tensor* out_marginals = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, marg_shape,
                                                         &out_marginals));
        
        // Compute marginals
        auto marginals = out_marginals->tensor<float, 2>();
        for (int i = 0; i < seq_length; i++){
            for (int j = 0; j < n_vars; j++){
                marginals(i, j) = forward_sp(i + 1, j) + backward_sp(i + 1, j);
                aux_array[j] = marginals(i, j);
            }
            float log_norm = LogSumExpP(aux_array, n_vars);
            for (int j = 0; j < n_vars; j++)
                marginals(i, j) = exp(marginals(i, j) - log_norm);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ChainCRF").Device(DEVICE_CPU), ChainCRFOp);
