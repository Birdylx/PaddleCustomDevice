# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
INDEX_ADD_CASE = [
    {"x_shape": [3, 2], "x_dtype": np.float32, "index_size": 2, "index_dtype": np.int32, "add_value_shape": [2, 2], "axis": 0},
    {"x_shape": [3, 2], "x_dtype": np.float32, "index_size": 2, "index_dtype": np.int64, "add_value_shape": [3, 2], "axis": 1},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "index_size": 6, "index_dtype": np.int32, "add_value_shape": [2, 6, 4], "axis": 1},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "index_size": 5, "index_dtype": np.int64, "add_value_shape": [2, 3, 5], "axis": 2},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float32, "index_size": 6, "index_dtype": np.int32, "add_value_shape": [2, 3, 6, 5], "axis": -2},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float32, "index_size": 5, "index_dtype": np.int64, "add_value_shape": [2, 5, 4, 5], "axis": 1},

    {"x_shape": [3, 2], "x_dtype": np.float16, "index_size": 2, "index_dtype": np.int32, "add_value_shape": [2, 2], "axis": 0},
    {"x_shape": [3, 2], "x_dtype": np.float16, "index_size": 2, "index_dtype": np.int64, "add_value_shape": [3, 2], "axis": 1},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "index_size": 6, "index_dtype": np.int32, "add_value_shape": [2, 6, 4], "axis": 1},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "index_size": 5, "index_dtype": np.int64, "add_value_shape": [2, 3, 5], "axis": 2},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float16, "index_size": 6, "index_dtype": np.int32, "add_value_shape": [2, 3, 6, 5], "axis": -2},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float16, "index_size": 5, "index_dtype": np.int64, "add_value_shape": [2, 5, 4, 5], "axis": 1},

    {"x_shape": [3, 2], "x_dtype": np.int32, "index_size": 2, "index_dtype": np.int32, "add_value_shape": [2, 2], "axis": 0},
    {"x_shape": [3, 2], "x_dtype": np.int32, "index_size": 2, "index_dtype": np.int64, "add_value_shape": [3, 2], "axis": 1},
    {"x_shape": [2, 3, 4], "x_dtype": np.int32, "index_size": 6, "index_dtype": np.int32, "add_value_shape": [2, 6, 4], "axis": 1},
    {"x_shape": [2, 3, 4], "x_dtype": np.int32, "index_size": 5, "index_dtype": np.int64, "add_value_shape": [2, 3, 5], "axis": 2},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.int32, "index_size": 6, "index_dtype": np.int32, "add_value_shape": [2, 3, 6, 5], "axis": -2},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.int32, "index_size": 5, "index_dtype": np.int64, "add_value_shape": [2, 5, 4, 5], "axis": 1},

]
# fmt: on


@ddt
class TestIndexAdd(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 2]
        self.x_dtype = np.float32
        self.index_size = 2
        self.index_dtype = np.int32
        self.add_value_shape = [2, 2]
        self.axis = 0

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        self.data_index = np.random.randint(
            low=0, high=self.x_shape[self.axis], size=self.index_size
        )
        self.data_add_value = self.generate_data(self.add_value_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        index = paddle.to_tensor(self.data_index, dtype=self.index_dtype)
        add_value = paddle.to_tensor(self.data_add_value, dtype=self.x_dtype)
        return paddle.index_add(x, index, self.axis, add_value)

    def index_add_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        index = paddle.to_tensor(self.data_index, dtype=self.index_dtype)
        add_value = paddle.to_tensor(self.data_add_value, dtype="float32")
        return paddle.index_add(x, index, self.axis, add_value).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.index_add_cast()
        return out

    @data(*INDEX_ADD_CASE)
    @unpack
    def test_check_output(
        self, x_shape, x_dtype, index_size, index_dtype, add_value_shape, axis
    ):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.index_size = index_size
        self.index_dtype = index_dtype
        self.add_value_shape = add_value_shape
        self.axis = axis
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()