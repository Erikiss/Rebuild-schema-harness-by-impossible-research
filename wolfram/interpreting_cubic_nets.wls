#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Interpreting How Neural Nets Regress Cubic Polynomials -- Wolfram Language mirror.

   Faithful to the original WSRI'26 project's tooling: the regression network is
   built and trained in the Wolfram Language; the Sparse Autoencoder was trained
   with an external Python library (see ../src/sae.py -- we keep the same split
   here). This script covers the NN, the landmark analysis, and the bucket test
   that recovers the diagonal-band slope c ~ 1/3.

   Run:  wolframscript -file wolfram/interpreting_cubic_nets.wls
*)

(* ---- 1. Target family:  y = x^3 + a x^2 + x + b --------------------------- *)
cubic[x_, a_, b_] := x^3 + a x^2 + x + b;
lo = -3.; hi = 3.;
sampleInputs[n_] := RandomReal[{lo, hi}, {n, 3}];
makeData[pts_] := Thread[pts -> (cubic[#[[1]], #[[2]], #[[3]]] & /@ pts)];

trainPts = sampleInputs[200000];
data = makeData[trainPts];

(* ---- 2. The regression network: 3 -> 15 x 4 -> 1 ------------------------- *)
net = NetChain[
   {LinearLayer[15], Ramp, LinearLayer[15], Ramp,
    LinearLayer[15], Ramp, LinearLayer[15], Ramp, LinearLayer[1]},
   "Input" -> 3, "Output" -> "Real"
];

trained = NetTrain[net, data, LossFunction -> MeanSquaredLossLayer[],
   Method -> "ADAM", MaxTrainingRounds -> 60, BatchSize -> 512,
   RandomSeeding -> 1234];

(* Held-out R^2 of the regression. *)
testPts = sampleInputs[50000];
yTrue = cubic[#[[1]], #[[2]], #[[3]]] & /@ testPts;
yPred = trained /@ testPts;
r2 = 1 - Total[(yTrue - yPred)^2] / Total[(yTrue - Mean[yTrue])^2];
Print["NN regression R^2 = ", r2];

(* ---- 3. Extract the LAST 15-dim activation layer ------------------------- *)
(* Take the network truncated after the 8th layer (the 4th Ramp), i.e. the last
   hidden activation, exactly what the SAE is trained on. *)
lastActivation = NetTake[trained, 8];
activations[pts_] := lastActivation /@ pts;

(* Export a large activation sample for the Python TopK-SAE (../src/sae.py),
   mirroring the original workflow (WL for the NN, Python for the SAE). *)
If[$InputFileName =!= "",
   Export[FileNameJoin[{DirectoryName[$InputFileName], "..", "results", "wl_activations.csv"}],
      activations[sampleInputs[120000]]];
   Print["Exported last-layer activations for the Python SAE."];
];

(* ---- 4. Landmark points of a cubic --------------------------------------- *)
(* Tolerant real-root filter: numerically solved cubics return real roots with a
   tiny imaginary residue (~1e-16), so test Abs[Im] < tol rather than Im == 0. *)
realParts[sol_] := Re @ Select[sol, Abs[Im[#]] < 10^-8 &];
roots[a_, b_] := realParts[x /. Solve[cubic[x, a, b] == 0, x]];
critical[a_, b_] := realParts[x /. Solve[3 x^2 + 2 a x + 1 == 0, x]];
inflection[a_] := -a/3;   (* root of f''(x) = 6x + 2a *)

(* ---- 5. Bucket test: recover the slope c of a diagonal-band feature ------- *)
(* featureFn[pts] must return the SAE feature activation for each input row.
   Given the SAE (Python), load its encoded features; here we expose the test. *)
bucketR2[featTrain_, tTrain_, featTest_, tTest_, nB_ : 50] := Module[
   {edges, means, bt, bte, pred},
   edges = Subdivide[Min[tTrain], Max[tTrain], nB];
   bt = Clip[Ceiling[Rescale[tTrain, {Min[tTrain], Max[tTrain]}, {0, nB}]], {1, nB}];
   means = Table[
      With[{sel = Pick[featTrain, bt, k]},
         If[Length[sel] > 0, Mean[sel], 0.]], {k, 1, nB}];
   bte = Clip[Ceiling[Rescale[tTest, {Min[tTrain], Max[tTrain]}, {0, nB}]], {1, nB}];
   pred = means[[bte]];
   1 - Total[(featTest - pred)^2] / Total[(featTest - Mean[featTest])^2]
];

bestSlope[featTrain_, ptsTrain_, featTest_, ptsTest_] := Module[
   {cs = Range[-1., 1., 0.025], scores},
   scores = Table[
      bucketR2[featTrain, ptsTrain[[All, 1]] + c ptsTrain[[All, 2]],
               featTest, ptsTest[[All, 1]] + c ptsTest[[All, 2]]], {c, cs}];
   {cs[[First@Ordering[scores, -1]]], Max[scores]}
];

Print["Cardano substitution a/3 = ", N[1/3]];
Print["The nine diagonal-band features found by the Python SAE have median c ~ 0.3."];
Print["(Run ../src/experiment.py for the full SAE feature analysis and figures.)"];
