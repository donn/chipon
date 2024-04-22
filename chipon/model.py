from typing import List

import numpy as np
import torch

import torch.nn as nn
import torch.ao.nn.quantized as qnn

from . import layers
from .constants import test_bench_template


class Model:
    def __init__(self, model: nn.Sequential):
        self.model = model
        self.layers = []

    def __str__(self):
        return '\n'.join(str(layer) for layer in self.layers)

    def parse_layers(self):
        input_scale = None
        #output_scale = None
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                print(f"Adding linear layer {layer}")
                self.layers.append(layers.Linear.from_pytorch_layer(layer, i))
            elif isinstance(layer, qnn.Linear):
                print(f"Adding quantized linear layer {layer}")
                self.layers.append(layers.Linear.from_quantized_pytorch_layer(layer, i, input_scale))
            elif isinstance(layer, nn.ReLU):
                print(f"Adding ReLU layer {layer}")
                self.layers.append(layers.ReLU(self.model[i - 1].out_features, i))
            elif isinstance(layer, qnn.Quantize):
                input_scale = layer.scale[0]
            elif isinstance(layer, qnn.DeQuantize):
                pass # output_scale = layer.scale[0]
            else:
                raise ValueError(f'Unknown layer type {layer}')

    def forward_range(self, ranges: List[List[float]]):
        start = np.array(ranges)

        for layer in self.layers:
            start = layer.forward_range(start)

    def get_vars(self):
        in_params = [f"in{i}" for i in range(self.layers[0].shape[0])]
        out_params = [f"out{i}" for i in range(self.layers[-1].shape[-1])]

        in_definitions = [f"    input [{self.layers[0].in_bits[i] - 1}:0] {in_params[i]};"
                          for i in range(self.layers[0].shape[0])]

        out_definitions = [f"    output [{self.layers[-1].out_bits[i] - 1}:0] {out_params[i]};"
                           for i in range(self.layers[-1].shape[-1])]

        return in_params, out_params, in_definitions, out_definitions

    def emit(self):
        out = ["`timescale 1ns / 1ps"]

        in_params, out_params, in_definitions, out_definitions = self.get_vars()

        top = [
            f"module top({','.join(in_params)}, {','.join(out_params)});",
            *in_definitions,
            *out_definitions,
        ]

        in_wires = in_params
        out_wires = []

        for i, layer in enumerate(self.layers):
            out.append(layer.emit())

            out_wires = []
            for j in range(layer.shape[-1]):
                top.append(f"    wire [{layer.out_bits[j]}:0] layer_{i}_out_{j};")
                out_wires.append(f"layer_{i}_out_{j}")

            top.append(f"    {layer.name} layer_{i}({','.join(in_wires)}, {','.join(out_wires)});")

            in_wires = out_wires

        assigns = [f"    assign out{i} = {out_wire};" for i, out_wire in enumerate(out_wires)]

        top.extend(assigns)
        top.append("endmodule")

        out.append('\n'.join(top))

        return '\n'.join(out)

    def emit_test_bench(self):
        in_params, out_params, in_definitions, out_definitions = self.get_vars()

        assigns = [f"        assign {i} = 0;" for i in in_params]

        return test_bench_template.format(
            in_params=', '.join(in_params),
            out_params=', '.join(out_params),
            in_definitions='\n    '.join(in_definitions),
            out_definitions='\n    '.join(out_definitions),
            assignments='\n'.join(assigns),
        )


def test():
    simple_model = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
    )

    simple_model[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    simple_model[0].bias = nn.Parameter(torch.tensor([1.0]))

    model = Model(simple_model)
    model.parse_layers()
    model.forward_range([[1.0, 100.0], [0.0, 1024.0]])

    print(model)
    code = model.emit()

    with open('test.v', 'w') as f:
        f.write(code)

    with open('test_tb.v', 'w') as f:
        f.write(model.emit_test_bench())


if __name__ == '__main__':
    test()
