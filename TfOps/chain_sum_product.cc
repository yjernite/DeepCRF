#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("ChainSumProduct")
    .Input("log_potentials: float32")
    .Input("tags: int32")
    .Output("forward_sp: float32")
    .Output("backward_sp: float32")
    .Output("gradients: float32")
    .Doc(R"doc(
    A module which helps with inference on a chain CRF (or MRF) by 
    implementing Sum-Product message passing. Also computes the partition
    function gradients.
    )doc");

using namespace tensorflow;

float LogSumExpDP(float nums[], size_t ct) {
	float max_exp = nums[0], sum = 0.0;
	size_t i;
	for (i = 1 ; i < ct ; i++)
		if (nums[i] > max_exp)
			max_exp = nums[i];
	for (i = 0; i < ct ; i++)
		sum += exp(nums[i] - max_exp);
	return log(sum) + max_exp;
}

class ChainSumProductOp : public OpKernel {
    public: explicit ChainSumProductOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& log_potentials = context->input(0);
        auto log_pots = log_potentials.tensor<float, 3>(); // TODO: replace 3
        
        const Tensor& in_tags = context->input(1);
        auto tags = in_tags.tensor<int, 1>();
        
        int seq_length = log_potentials.dim_size(0);
        int n_vars = log_potentials.dim_size(1);
        
        float aux_array[n_vars];
        float aux_array_grad[n_vars * n_vars];
        
        // prepare TensorShape vectors
        TensorShape dyn_shape;
        dyn_shape.AddDim(seq_length + 1);
        for (int i = 1; i < log_potentials.dims() - 1; i ++)
            dyn_shape.AddDim(n_vars);
        
        // Create an output tensor: forward_sp
        Tensor* out_forward_sp = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, dyn_shape,
                                                         &out_forward_sp));
        
        // Compute forward_sp
        auto forward_sp = out_forward_sp->tensor<float, 2>(); // TODO: replace 2
        for (int j =0; j < n_vars; j++){
            forward_sp(0, j) = 0;
        }
        for (int i = 0; i < seq_length; i++){
            for (int k =0; k < n_vars; k++){
                for (int j =0; j < n_vars; j++){
                    aux_array[j] = forward_sp(i, j) + log_pots(i, j, k);
                }
                forward_sp(i + 1, k) = LogSumExpDP(aux_array, n_vars);
            }
        }
        
        // Create an output tensor: backward_sp
        Tensor* out_backward_sp = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, dyn_shape,
                                                         &out_backward_sp));
        
        // Compute backward_sp
        auto backward_sp = out_backward_sp->tensor<float, 2>();
        for (int j =0; j < n_vars; j++){
            backward_sp(seq_length, j) = 0;
        }
        for (int i = seq_length - 1; i >= 0; i--){
            for (int j =0; j < n_vars; j++){
                for (int k =0; k < n_vars; k++){
                    aux_array[k] = backward_sp(i + 1, k) + log_pots(i, j, k);
                }
                backward_sp(i, j) = LogSumExpDP(aux_array, n_vars);
            }
        }
        
        // Create an output tensor: gradients
        Tensor* out_gradients = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, log_potentials.shape(),
                                                         &out_gradients));
        
        // Compute gradients. TODO: deal with first one
        auto gradients = out_gradients->tensor<float, 3>();
        for (int j = 0; j < n_vars; j++)
            for (int k = 0; k < n_vars; k++)
                gradients(0, j, k) = 0;
        for (int i = 1; i < seq_length; i++){
            for (int j = 0; j < n_vars; j++){
                for (int k = 0; k < n_vars; k++){
                    gradients(i, j, k) = forward_sp(i, j) + log_pots(i, j, k) + backward_sp(i + 1, k);
                    aux_array_grad[j * n_vars + k] = gradients(i, j, k);
                }
            }
            float log_norm = LogSumExpDP(aux_array_grad, n_vars * n_vars);
            int mask = (tags(i-1) + tags(i)) > 0;
            for (int j = 0; j < n_vars; j++){
                for (int k = 0; k < n_vars; k++){
                    gradients(i, j, k) = -mask * exp(gradients(i, j, k) - log_norm);
                }
            }
            gradients(i, tags(i-1), tags(i)) += mask;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ChainSumProduct").Device(DEVICE_CPU), ChainSumProductOp);
