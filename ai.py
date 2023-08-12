import dataclasses
from scipy.special import expit
import random
import math
@dataclasses.dataclass()
class nn:
    layers: int
    input_nodes: int
    output_nodes: int
    layer_nodes: int
    @dataclasses.dataclass()
    class node:
        weights: list
        bias: int
    def get_node_list(self, number=None):
        def get_number(count):
            if number:
                return [number]*count
            return [random.random() for _ in range(count)]
        out = [get_number(self.input_nodes)]
        for _ in range(self.layers):
            out.append(get_number(self.layer_nodes))
        out.append(get_number(self.output_nodes))
        for i in range(len(out)):
            for j in range(len(out[i])):
                if i == len(out)-1:
                    out[i][j] = self.node([], get_number(1)[0])
                else:
                    out[i][j] = self.node(get_number(len(out[i+1])), get_number(1)[0])
        self.node_list = out
        return out
    def calculate_nodes(self, input_node_list, node_list=None):
        if node_list == None:
            node_list = self.node_list
        for i in range(len(node_list[0])):
            node_list[0][i].value = input_node_list[i]
        for i in range(len(node_list)):
            for j in range(len(node_list[i])):
                node = node_list[i][j]
                weights_times_values = []
                if i > 0:
                    for weight_node in node_list[i-1]:
                        weights_times_values.append(weight_node.value*weight_node.weights[j])
                node_list[i][j].value = expit(sum(weights_times_values)+node.bias)
                
        return node_list
    
a = nn(2, 3, 1, 4)
a.get_node_list()
for i in a.calculate_nodes([0.0, 0.0, 0.0]):
    print([j.value for j in i])
