#
# Copyright 2016 D-Wave Systems Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
The main logic for the Beartooth demo.

The idea of this demo is to provide a visual to reinforce the 'sampling as
water flowing to low points in a landsape' description of the D-Wave computer.

The landscape is a discrete nxn grid of altitudes, f(x, y); we will construct
a QUBO problem where samples that encode locations take on the energy equal to
the altitude at that location.
    (and samples that do not represent valid locations have higher energy than...)

We chose a unary encoding for coordinates.  Thus each coordinate will need
n - 1 bits, x0, x1, ..., xn-2, and y0, y1, ..., yn-2.  E.g. on a 4x4 grid:
    x   encoded
    0   000
    1   100
    2   110
    3   111
So the QUBO will have 2 * (n - 1) variables.  To set the energy of the samples
to be the altitudes, we make use of the unary encoding to telescope the
energies:
    for i in range(0, n-1):
        Q[xi, xi] = f(i + 1, 0) - f(i, 0)
        Q[yi, yi] = f(0, i + 1) - f(0, i)
    for i in range(0, n-1):
        for j in range(0, n-1):
            Q[xi, yj] = f(i + 1, j + 1) - f(i, j + 1)
                      - f(i, j + 1) + f(i, j)
And add the constant offset f(0, 0), which we basically ignore in this code.

To bias the QUBO to return valid coordinates, we add the following:
    for i in range(1, n-1):
        Q[xi-1,xi] = -ENCODING_BIAS
        Q[xi, xi] = ENCODING_BIAS
This will add at least ENCODING_BIAS to the energy for invalid samples.
"""

from collections import defaultdict

import dwave_sapi2.core
import dwave_sapi2.remote
import dwave_sapi2.util
import dwave_sapi2.embedding


ENCODING_BIAS = 5

# f(x, y)
LANDSCAPE = [
        [ 4, 3, 2, 2, 3, 2, 3, 4, 4, 5 ],
        [ 3, 2, 1, 2, 2, 2, 2, 3, 4, 6 ],
        [ 3, 2, 0, 1, 2, 2, 3, 3, 5, 7 ],
        [ 3, 2, 1, 1, 2, 2, 3, 5, 7, 8 ],
        [ 3, 2, 1, 2, 3, 4, 4, 6, 7, 7 ],
        [ 2, 2, 3, 4, 4, 5, 6, 7, 6, 5 ],
        [ 2, 3, 3, 4, 5, 7, 7, 6, 5, 4 ],
        [ 4, 5, 5, 6, 6, 9, 8, 7, 5, 4 ],
        [ 5, 6, 6, 7, 8, 8, 7, 7, 6, 5 ],
        [ 7, 7, 8, 9, 9, 9, 8, 8, 6, 5 ],
    ]

EMBEDDING = [
       [122, 218, 222, 314], [113, 209, 305, 309, 317], [104, 200, 296, 301], [107, 203, 205, 299], [114, 210, 213, 306], [115, 211, 307], [123, 215, 219, 223, 315], [105, 201, 207, 297], [106, 202, 298], [302, 310, 318], [300, 308, 312, 316], [204, 212, 216, 220], [206, 208, 214], [111, 112, 119, 127], [108, 116, 120, 124], [110, 118, 126], [109, 117, 121, 125], [217, 303, 311, 313, 319]
]


def _x_shift(x):
    return LANDSCAPE[x + 1][0] - LANDSCAPE[x][0]

def _y_shift(y):
    return LANDSCAPE[0][y + 1] - LANDSCAPE[0][y]

def _angle_shift(x, y):
    return (  LANDSCAPE[x + 1][y + 1] - LANDSCAPE[x][y + 1]
            - LANDSCAPE[x + 1][y]     + LANDSCAPE[x][y])

def _get_qubo():
    """
    Get the qubo for the landscape.  See module docstring for details.
    """
    # variables: [ x0 ... xn-2 y0 ... yn-2 ]
    num_vars = len(LANDSCAPE) - 1
    x_vars = list(range(0, num_vars))
    y_vars = list(range(num_vars, 2*num_vars))

    Q = {
        (x_vars[x], x_vars[x]): ENCODING_BIAS + _x_shift(x)
            for x in range(1, num_vars)
    }
    Q[x_vars[0], x_vars[0]] = _x_shift(0)
    Q.update({
        (y_vars[y], y_vars[y]): ENCODING_BIAS + _y_shift(y)
            for y in range(1, num_vars)
    })
    Q[y_vars[0], y_vars[0]] = _y_shift(0)

    Q.update({
        (x_vars[x], y_vars[y]): _angle_shift(x, y)
            for x in range(num_vars) for y in range(num_vars)
    })

    Q.update({
        (x_vars[x], x_vars[x - 1]): -ENCODING_BIAS
            for x in range(1, num_vars)
    })
    Q.update({
        (y_vars[y], y_vars[y - 1]): -ENCODING_BIAS
            for y in range(1, num_vars)
    })
    # We don't handle the constant offset: LANDSCAPE[0][0]

    return Q


def _parse_coord(sample, variables):
    """
    See if the variables in the sample are a valid encoding of a coordinate.

    The encoding is defined this way:
        For n-1 variables, the axis' range is {0, 1, ..., n}.
        A coordinate with value x is encoded as
            v_i = 0 for x <= i < n - 1
            v_i = 1 for 0 <= i < x
    """
    coord = 0
    for v, v_index in enumerate(variables):
        if sample[v_index] == 1:
            coord = v + 1
            if v != 0 and sample[v_index - 1] == 0:
                # Not a valid encoding
                return None
    return coord


def _interpret_samples(samples):
    """
    Takes a list of (sample, num_occurrences) and returns a mapping from
    (x, y) coordinates on the landscape to the number of time it was sampled.

    The mapping is a defaultdict, so you can look up any coordinate, even if
    it was never sampled.
    """
    interpreted_samples = defaultdict(int)

    num_vars = len(LANDSCAPE) - 1
    # variables: [ x0 ... xn-2 y0 ... yn-2 ]
    x_vars = list(range(0, num_vars))
    y_vars = list(range(num_vars, 2*num_vars))

    def _interpret_sample(sample):
        """
        Place num_occurrences into interpreted_samples for valid samples.
        """
        s, num = sample
        x = _parse_coord(s, x_vars)
        if x is None:
            return
        y = _parse_coord(s, y_vars)
        if y is None:
            return
        # '+=' since some samples could be duplicates:
        interpreted_samples[x, y] += num

    # This is being done for the side effects of _interpret_sample on
    # the interpreted_samples dict.
    map(_interpret_sample, samples)

    return interpreted_samples


def get_samples(solver, num_reads):
    """
    Return a coordinate -> number of samples mapping.
    """
    Q = _get_qubo()
    h, J, _energy_offset = dwave_sapi2.util.qubo_to_ising(Q)

    eh, eJ, chain_J, _embedding = dwave_sapi2.embedding.embed_problem(
            h, J, EMBEDDING, dwave_sapi2.util.get_hardware_adjacency(solver))
    eJ.update(chain_J)

    answers = dwave_sapi2.core.solve_ising(solver, eh, eJ, num_reads=num_reads)

    # This could cause some solutions to become duplicates:
    unembedded_solutions = dwave_sapi2.embedding.unembed_answer(answers['solutions'], EMBEDDING, h=eh, j=eJ)

    samples = zip(unembedded_solutions, answers['num_occurrences'])
    interpreted_samples = _interpret_samples(samples)
    return interpreted_samples


# Simple script to just fetch 100 samples and print them in a grid next to the original landscape.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Beartooth Demo')
    parser.add_argument('url', help='The SAPI url.')
    parser.add_argument('apitoken', help='Your SAPI apitoken.')
    parser.add_argument('solver', help='The name of the solver you wish to use.')
    args = parser.parse_args()

    connection = dwave_sapi2.remote.RemoteConnection(args.url, args.apitoken)
    solver = connection.get_solver(args.solver)

    samples = get_samples(solver, 100)

    print
    print 'Landscape:'
    for i in LANDSCAPE:
        for j in i:
            print '%2d' % j,
        print

    print
    print 'Samples:'
    n = len(LANDSCAPE)
    for i in range(n):
        for j in range(n):
            print '%2d' % samples[i,j],
        print

