import tensorflow as tf

class ChainCRFTest(tf.test.TestCase):
    def testChainCRF(self):
        with self.test_session():
            pre_pots = [[[1, 3, 1], [4, 1, 2], [2, 1, 1]],
                        [[2, 2, 1], [3, 1, 1], [1, 2, 3]],
                        [[3, 1, 1], [2, 1, 2], [1, 3, 2]],
                        [[1, 2, 1], [4, 1, 3], [1, 1, 1]]]
            potentials = tf.convert_to_tensor(pre_pots, tf.float32)
            forward_sp, backward_sp, gradients = tf.user_ops.chain_sum_product(potentials)
            partition, marginals = tf.user_ops.chain_crf(potentials, forward_sp, backward_sp, gradients)
            print(partition.eval()) # TODO


# bazel test tensorflow/python:chain_crf_op_test --verbose_failures
