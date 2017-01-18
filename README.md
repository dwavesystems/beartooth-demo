# Beartooth demo

The idea of this demo is to provide a visual to reinforce the 'sampling as
water flowing to low points in a landsape' description of the D-Wave computer.

## Details

The landscape is a discrete nxn grid of altitudes, f(x, y); we will construct
a QUBO problem where samples that encode locations take on the energy equal to
the altitude at that location.
    (and samples that do not represent valid locations have higher energy than...)

We chose a unary encoding for coordinates.  Thus each coordinate will need
n - 1 bits, x0, x1, ..., xn-2, and y0, y1, ..., yn-2.  E.g. on a 4x4 grid:

<pre>
  x   encoded
  0   000
  1   100
  2   110
  3   111
</pre>

So the QUBO will have 2 * (n - 1) variables.  To set the energy of the samples
to be the altitudes, we make use of the unary encoding to telescope the
energies:

```python
for i in range(0, n-1):
    Q[xi, xi] = f(i + 1, 0) - f(i, 0)
    Q[yi, yi] = f(0, i + 1) - f(0, i)
for i in range(0, n-1):
    for j in range(0, n-1):
        Q[xi, yj] = f(i + 1, j + 1) - f(i, j + 1) - f(i, j + 1) + f(i, j)
```

And add the constant offset f(0, 0), which we basically ignore in this code.
To bias the QUBO to return valid coordinates, we add the following:

```python
for i in range(1, n-1):
    Q[xi-1,xi] = -ENCODING_BIAS
    Q[xi, xi] = ENCODING_BIAS
```
This will add at least `ENCODING_BIAS` to the energy for invalid samples.
