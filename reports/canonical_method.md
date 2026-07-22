# Canonical Geometric Pedestrian Model

## Purpose

This formulation preserves the manuscript's intended social-force-like dynamics and its
spacetime visualisation while removing the inconsistent claim that velocity is repeatedly
reset inside a second-order ODE solver.

## Agent-specific potential

For pedestrian `i` at position `x`, define

```text
U_i(x,t) = A_d ||x-d_i||
         + sum_j A_p exp(-( ||x-x_j||-2R )/B_p)
         + sum_k A_o exp(-( ||x-o_k||-R )/B_o).
```

The destination term is linear distance. This is the form required by the parameter units,
the positive attractive coefficient, the desired-speed expression in Section 6.1, and the
reported simulation trajectory. The reciprocal destination term printed in Equation 23 is
not consistent with those results.

The social acceleration field is

```text
a_i(x,t) = -grad U_i(x,t).
```

## Geometric representation

An effective static metric can be written as

```text
ds_i^2 = -(c_ref^2 + 2 U_i(x,t)) dt^2 + dx^2 + dy^2.
```

In the weak-field, low-speed limit, its spatial geodesic equation produces

```text
d^2 x_i / dt^2 = -grad U_i.
```

This is a mathematical representation of the behavioural field. `c_ref` is a reference
scale; using physical light speed does not imply that relativistic effects influence walking.

## Behavioural update

The submitted simulation is reproduced by the overdamped or instantaneous-decision limit

```text
v_i* = -kappa grad U_i,
x_i(t+dt) = x_i(t) + dt v_i*,
```

with `kappa=0.1 s`. A maximum walking speed can be imposed after constructing `v_i*`.

Equivalently, a finite response-time extension is

```text
tau_v dv_i/dt = v_i* - v_i.
```

The limit `tau_v -> 0` recovers the implemented model. This makes its relationship to the
Social Force Model explicit: both use relaxation toward a behaviourally determined desired
velocity, while here the desired velocity is obtained from an agent-specific geometric field.

## Spacetime visualisation

With physical reference speed, the raw weak-field time-factor deviation is approximately

```text
Delta f_i = f_i - 1 ~= U_i / c_ref^2,
```

which is around `1e-17` to `1e-15` in the reconstructed example. It must not be presented as
a measurable psychological time dilation.

For legible figures, define a dimensionless visual elevation

```text
H_i = clip((U_i-Q_0.02(U_i)) / (Q_0.98(U_i)-Q_0.02(U_i)), 0, 1).
```

Every figure must label `H_i` as a rescaled potential elevation and report the raw `Delta f_i`
range. This preserves the spacetime visualisation while making the amplification transparent.

## Verified reconstruction

Against the manuscript's Table 2:

- trajectory coordinate RMSE: 0.169 m;
- speed RMSE: 0.453 m/s;
- minimum two-agent separation: 0.670 m;
- maximum speed: below the configured 1.5 m/s cap.

The remaining mismatch should not be hand-tuned against Table 2 because Table 2 is simulated,
not empirical. Subsequent calibration must use the frozen Jülich training runs.

