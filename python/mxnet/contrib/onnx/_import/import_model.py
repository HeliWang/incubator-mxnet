# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
"""import function"""
# pylint: disable=no-member

from __future__ import absolute_import
from import_onnx import GraphProto
from collections import namedtuple
import mxnet as mx
from onnx.backend.base import Backend
#from import_onnx import GraphProto
#import backend_rep as MXNetBackendRep
import unittest
import numpy as np
import numpy.testing as npt
from onnx import helper

def import_model(model_file):
    """Imports the ONNX model file passed as a parameter into MXNet symbol and parameters.

    Parameters
    ----------
    model_file : str
        ONNX model file name

    Returns
    -------
    Mxnet symbol and parameter objects.

    sym : mxnet.symbol
        Mxnet symbol
    params : dict of str to mx.ndarray
        Dict of converted parameters stored in mxnet.ndarray format
    """
    graph = GraphProto()

    # loads model file and returns ONNX protobuf object
    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed")
    model_proto = onnx.load(model_file)
    sym, params = graph.from_onnx(model_proto.graph)
    return sym, params

def getRandom(shape):
    return np.random.ranf(shape).astype("float32")


def verify_onnx_forward_impl(sym, params, input_data, output_data):
    """Verifies result after inference"""
    print("Converting onnx format to mxnet's symbol and params...")
    sym, params = sym,params

    # create module
    mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('input_0', input_data.shape)], label_shapes=None)
    mod.set_params(arg_params=params, aux_params=params, allow_missing=True, allow_extra=True)
    # run inference
    Batch = namedtuple('Batch', ['data'])

    mod.forward(Batch([mx.nd.array(input_data)]), is_train=False)

    # Run the model with an onnx backend and verify the results
    npt.assert_equal(mod.get_outputs()[0].shape, output_data.shape)
    npt.assert_almost_equal(output_data, mod.get_outputs()[0].asnumpy(), decimal=3)
    print("Conversion Successful")


sym, params = import_model("/Users/aanirud/Code/scripts/onnxModels/inception_v2/model.onnx")
npz_path = '/Users/aanirud/Code/scripts/onnxModels/densenet121/test_data_0.npz'
sample = np.load(npz_path, encoding='bytes')
inputs = list(sample['inputs'])
outputs = list(sample['outputs'])
input_data = np.asarray(inputs[0], dtype=np.float32)#, (0,2,3,1))
output_data = np.asarray(outputs[0], dtype=np.float32)
verify_onnx_forward_impl(sym, params, input_data, output_data)

'''
node_def = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=0.5, beta=0.5)
A = getRandom([3,4])
B = getRandom([4,3])
C = getRandom([3,3])
print(A)
print(B)
print(C)
output = mxnet_backend.run_node(node_def, [A,B,C])[0]
#npt.assert_almost_equal(output, np.negative(ip1), decimal=5)
print("Passed GEMM")
'''