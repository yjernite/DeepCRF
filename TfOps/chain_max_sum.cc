#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("ChainMaxSum")
    .Input("log_potentials: float32")
    .Input("tags: int32")
    .Output("forward_ms: float32")
    .Output("backward_ms: int32")
    .Output("tagging: float32")
    .Doc(R"doc(
    A module which helps with inference on a chain CRF (or MRF) by 
    implementing Max-Sum message passing. Also computes the partition
    function gradients.
    )doc");

using namespace tensorflow;

float MaxMS(float nums[], size_t ct, float &max, int &max_id) {
	max = nums[0];
	max_id = 0;
	size_t i;
	for (i = 1 ; i < ct ; i++)
		if (nums[i] > max){
		    max = nums[i];
            max_id = i;
		}
	return max;
}

class ChainMaxSumOp : public OpKernel {
    public: explicit ChainMaxSumOp(OpKernelConstruction* context) : OpKernel(context) {
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
                
        // prepare TensorShape vectors
        TensorShape dyn_shape;
        dyn_shape.AddDim(seq_length + 1);
        for (int i = 1; i < log_potentials.dims() - 1; i ++)
            dyn_shape.AddDim(n_vars);
        
        // Create output tensors
        Tensor* out_forward_ms = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, dyn_shape,
                                                         &out_forward_ms));
        Tensor* out_backward_ms = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, dyn_shape,
                                                         &out_backward_ms));
        
        // Compute forward_ms
        auto forward_ms = out_forward_ms->tensor<float, 2>();
        auto backward_ms = out_backward_ms->tensor<int, 2>();
        for (int j =0; j < n_vars; j++){
            forward_ms(0, j) = 0;
        }
        for (int i = 0; i < seq_length; i++){
            for (int k =0; k < n_vars; k++){
                float max;
                int max_id;
                for (int j =0; j < n_vars; j++){
                    aux_array[j] = forward_ms(i, j) + log_pots(i, j, k);
                }
                MaxMS(aux_array, n_vars, max, max_id);
                forward_ms(i + 1, k) = max;
                backward_ms(i + 1, k) = max_id;
            }
        }
        
        
        // Create an output tensor: tagging
        TensorShape tagging_shape;
        tagging_shape.AddDim(seq_length);
        tagging_shape.AddDim(n_vars);
        Tensor* out_tagging = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, tagging_shape,
                                                         &out_tagging));
        
        // Compute tagging.
        auto tagging = out_tagging->tensor<float, 2>();
        int current = 0;
        for (int i = seq_length - 1; i >= 0; i--){
            for (int j =0; j < n_vars; j++)
                tagging(i, j) = 0;
            if (tags(i) == 0){
                tagging(i, 0) = 1;
                current = 0;
            }
            else{
                current = backward_ms(i + 1, current);
                tagging(i, current) = 1;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ChainMaxSum").Device(DEVICE_CPU), ChainMaxSumOp);
