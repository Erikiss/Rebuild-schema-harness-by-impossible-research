#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Interpreting How Neural Nets Regress Cubic Polynomials -- Wolfram Language mirror.

   This is the reference Wolfram Language version of the reproduction, matching the
   tooling used in the original WSRI'26 project. It is functionally identical to the
   NumPy code in ../src/: train a 1 -> H -> 1 Ramp network on a cubic, then read the
   piecewise-linear function straight off the weights.

   Run:  wolframscript -file wolfram/interpreting_cubic_nets.wls
*)

(* ---- 1. Target cubic and data --------------------------------------------- *)
a3 = 1.; a2 = 0.; a1 = -3.; a0 = 0.;
f[x_] := a3 x^3 + a2 x^2 + a1 x + a0;   (* f(x) = x^3 - 3x *)
fpp[x_] := 6 a3 x + 2 a2;               (* curvature f''(x) = 6x *)

domain = {-2., 2.};
nTrain = 400;
xTrain = Subdivide[domain[[1]], domain[[2]], nTrain - 1];
data = Thread[xTrain -> f /@ xTrain];

(* ---- 2. The network: 1 -> H (Ramp) -> 1 ----------------------------------- *)
hidden = 24;
net = NetChain[
   {LinearLayer[hidden], ElementwiseLayer[Ramp], LinearLayer[1]},
   "Input" -> "Real", "Output" -> "Real"
];

trained = NetTrain[
   net, data,
   LossFunction -> MeanSquaredLossLayer[],
   Method -> "ADAM",
   MaxTrainingRounds -> 4000,
   RandomSeeding -> 1234
];

(* ---- 3. Read the piecewise-linear function off the weights ----------------- *)
(* First layer: z_i = w_i x + b_i.   Second layer: y = v . h + c.               *)
w = Flatten @ NetExtract[trained, {1, "Weights"}];   (* length H *)
b = NetExtract[trained, {1, "Biases"}];              (* length H *)
v = Flatten @ NetExtract[trained, {3, "Weights"}];   (* length H *)
c = First @ NetExtract[trained, {3, "Biases"}];      (* scalar   *)

(* Breakpoint / delta-slope / orientation for every neuron.                     *)
breakpoints = -b/w;              (* beta_i = -b_i / w_i          *)
deltaSlopes = w v;               (* mu_i    = w_i v_i            *)
orientations = Sign[w];          (* s_i     = sign(w_i)          *)
slopeJumps = orientations deltaSlopes;   (* |w_i| v_i : slope change L->R *)

(* Analytic reconstruction directly from the weights (no call to `trained`).    *)
yPWL[x_] := c + Total[v Ramp[w x + b]];

(* ---- 4. The interpretation is EXACT --------------------------------------- *)
(* The net is scalar-in / scalar-out ("Input"/"Output" -> "Real"), so trained[x]
   returns a plain Real -- no First/@ unwrapping needed. *)
xDense = Subdivide[domain[[1]], domain[[2]], 2000];
maxDiff = Max @ Abs[(trained /@ xDense) - (yPWL /@ xDense)];
Print["Interpretation exactness  max|net - analytic CPWL| = ", ScientificForm[maxDiff]];

trainMSE = Mean[((trained /@ xTrain) - (f /@ xTrain))^2];
Print["Train MSE = ", ScientificForm[trainMSE]];

(* ---- 5. Kinks congregate where |f''| is large ----------------------------- *)
inside = Select[Transpose[{breakpoints, slopeJumps}],
   domain[[1]] < #[[1]] < domain[[2]] && Abs[#[[2]]] > 1.*^-3 &];
betas = inside[[All, 1]];
jumps = Abs[inside[[All, 2]]];
meanCurv = Mean[Abs[fpp /@ xDense]];
weightedCurv = Total[jumps Abs[fpp /@ betas]] / Total[jumps];
Print["Curvature enrichment at kinks = ", weightedCurv/meanCurv,
      "x  (>1 => kinks favour high-curvature regions)"];

(* ---- 6. Figures ----------------------------------------------------------- *)
fitPlot = Show[
   Plot[{f[x], yPWL[x]}, {x, domain[[1]], domain[[2]]},
      PlotLegends -> {"target f(x)=x^3-3x", "Ramp-net fit (piecewise linear)"},
      PlotStyle -> {Black, Red}],
   ListPlot[Thread[{betas, yPWL /@ betas}],
      PlotStyle -> Directive[Orange, PointSize[0.012]]],
   PlotLabel -> "A Ramp net regresses the cubic as a piecewise-linear function"
];

rampsPlot = Plot[Evaluate[Table[v[[i]] Ramp[w[[i]] x + b[[i]]], {i, hidden}]],
   {x, domain[[1]], domain[[2]]},
   PlotLabel -> "Each neuron contributes one ramp; their sum (+c) is the fit"];

curvaturePlot = Show[
   Plot[Abs[fpp[x]], {x, domain[[1]], domain[[2]]}, Filling -> Axis,
      PlotLabel -> "Kinks cluster where the cubic bends most"],
   ListPlot[Thread[{betas, Abs[fpp /@ betas]}],
      PlotStyle -> Directive[Orange, PointSize[0.012]]]
];

If[$ScriptCommandLine =!= {},
   Export[FileNameJoin[{DirectoryName[$InputFileName], "..", "figures", "wl_01_fit.png"}], fitPlot];
   Export[FileNameJoin[{DirectoryName[$InputFileName], "..", "figures", "wl_02_ramps.png"}], rampsPlot];
   Export[FileNameJoin[{DirectoryName[$InputFileName], "..", "figures", "wl_04_curvature.png"}], curvaturePlot];
   Print["Saved Wolfram figures to figures/wl_*.png"];
];
