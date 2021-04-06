from onnxruntime.datasets import get_example
import numpy as np
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_out(onnx_path, input_tensor):
    example_model = get_example(onnx_path)
    sess = onnxruntime.InferenceSession(example_model)
    onnx_out = np.array(sess.run(None, {"input": to_numpy(input_tensor)}))
    return onnx_out