
-- ==
-- entry: main
-- compiled input @ ../../data/sahara.in.gz
-- output @ ../../data/sahara.out.gz

let logplus (x: f32) : f32 =
  if x > (f32.exp 1)
  then f32.log x else 1

let partitionCos [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : ([n]t, i32) =
  let cs  = map p arr
  let tfs = map (\f -> if f then 1 else 0) cs
  let ffs = map (\f -> if f then 0 else 1) cs
  let isT = scan (+) 0 tfs
  let isF0= scan (+) 0 ffs

  let i   = last isT
  let isF = map (+i) isF0
  let inds= map3 (\c iT iF ->
                    if c then iT-1
                         else iF-1
                 ) cs isT isF
  let r = scatter (replicate n dummy) inds arr
  in  (r, i)

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX (k2p2: i32) (N: i32) (f: f32) : [k2p2][N]f32 =
  map (\ i ->
        map (\j ->
                if i == 0 then 1f32
                else if i == 1 then r32 j
                else let (i', j') = (r32 (i / 2), r32 j)
                     let angle = 2f32 * f32.pi * i' * j' / f
                     in  if i % 2 == 0 then f32.sin angle
                                       else f32.cos angle
            ) (1 ... N)
      ) (iota k2p2)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-blockparallelism                   --
---------------------------------------------------

  let gauss_jordan [nm] (n:i32) (A: *[nm]f32): [nm]f32 =
    let m = nm / n in
    loop A for i < n do
      let v1 = A[i]

      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            let x = unsafe (A[j] / v1) in
                                if k < n-1  -- Ap case
                                then unsafe ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
                                else x      -- irow case
                   ) (iota (n*m))
      in  scatter A (iota (n*m)) A'

  let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m = 2*n
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m) -- (row, col)
                          in  if j < n then unsafe ( A[i,j] )
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (iota (n*m))
    let Ap' = unflatten n m (gauss_jordan n Ap)

    -- Drop the identity matrix at the front.
    in Ap'[0:n,n:n * 2]
--------------------------------------------------
--------------------------------------------------

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\v x y -> if (f32.isnan v) then 0 else x*y) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
    map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------


entry bfast_1 (k2p2: i32) (N: i32) (f: f32) : [k2p2][N]f32 = mkX k2p2 N f

entry bfast_2 [m][N][k2p2] (X: [k2p2][N]f32) (Xt: [N][k2p2]f32)
                           (Y: [m][N]f32) (n: i32)
                           : [m][k2p2][k2p2]f32 =
  let Xh  = unsafe (X[:,:n])
  let Xth = unsafe (Xt[:n,:])
  let Yh  = unsafe (Y[:,:n])
  let Xsqr = map (matmul_filt Xh Xth) Yh
  in Xsqr

entry bfast_3 [m][k2p2] (Xsqr: [m][k2p2][k2p2]f32) : [m][k2p2][k2p2]f32 =
  let Xinv = map mat_inv Xsqr
  in Xinv

entry bfast_4a [m][N][k2p2] (X: [k2p2][N]f32) (Y: [m][N]f32) (n: i32)
                            : [m][k2p2]f32 =
  let Xh  = unsafe (X[:,:n])
  let Yh  = unsafe (Y[:,:n])
  let beta0 = map (matvecmul_row_filt Xh) Yh
  in beta0

entry bfast_4b [m][k2p2] (Xinv: [m][k2p2][k2p2]f32) (beta0: [m][k2p2]f32)
                         : [m][k2p2]f32 =
  let beta = map2 matvecmul_row Xinv beta0
  in beta

entry bfast_4c [m][N][k2p2] (Xt: [N][k2p2]f32) (beta: [m][k2p2]f32)
                            : [m][N]f32 =
  let y_preds = map (matvecmul_row Xt) beta
  in y_preds

entry bfast_5 [m][N] (Y: [m][N]f32) (y_preds: [m][N]f32)
                     : ([m]i32, [m][N]f32, [m][N]i32) =
  let (Nss, y_errors, val_indss) = unsafe ( unzip3 <|
    map2 (\y y_pred ->
            let y_error_all = zip y y_pred |>
                map (\(ye,yep) -> if !(f32.isnan ye)
                                  then ye-yep else f32.nan )
            let (tups, Ns) = zip2 y_error_all (iota N) |>
                partitionCos (\(ye,_) -> !(f32.isnan ye)) (0.0, 0)
            let (y_error, val_inds) = unzip tups
            in  (Ns, y_error, val_inds)
         ) Y y_preds )
  in (Nss, y_errors, val_indss)

entry bfast_6 [m][N] (Y: [m][N]f32) (y_errors: [m][N]f32) (n: i32) (k2p2: i32)
                     : ([m]i32, [m]f32) =
  let Yh  = unsafe (Y[:,:n])
  let (nss, sigmas) = unzip <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then unsafe y_error[i] else 0.0) (iota n)
                        |> map (\ a -> a*a ) |> reduce (+) 0.0
            let sigma = f32.sqrt ( sigma / (r32 (ns-k2p2)) )
            in  (ns, sigma)
         ) Yh y_errors
  in (nss, sigmas)

entry bfast_7a [m][N] (y_errors: [m][N]f32) (nss: [m]i32) (h: i32)
                      : [m]f32 =
  let MO_fsts = zip y_errors nss |>
    map (\(y_error, ns) ->
            map (\i -> unsafe y_error[i + ns-h+1]) (iota h)
            |> reduce (+) 0.0
        )
  in MO_fsts

entry bfast_7b (N: i32) (n: i32) (lam: f32) : []f32 = -- [N-n]f32
  let BOUND = map (\q -> let t   = n+1+q
                         let tmp = logplus ((r32 t) / (r32 n))
                         in  lam * (f32.sqrt tmp)
                  ) (0 ... N-n-1)
  in BOUND -- [N-n]f32

-- o is N-n
entry bfast_8 [m][N][o] (Nss: [m]i32) (nss: [m]i32) (sigmas: [m]f32)
                        (MO_fsts: [m]f32) (y_errors: [m][N]f32)
                        (val_indss: [m][N]i32) (BOUND: [o]f32)
                        (h: i32) (n: i32)
                        : [m][o]f32 =
    let breakss = zip (zip3 Nss nss sigmas) (zip3 MO_fsts y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma), (MO_fst,y_error,val_inds) ) ->
            let MO = map (\j -> if j >= Ns-ns then 0.0
                                else if j == 0 then MO_fst
                                else  unsafe (-y_error[ns-h+j] + y_error[ns+j])
                         ) (0 ... N-n-1) |> scan (+) 0.0
            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
            let val_inds' = map (\i ->  if i < Ns - ns
                                        then (unsafe val_inds[i+ns]) - n
                                        else -1
                                ) (0 ... N-n-1)
            let MO'' = scatter (replicate (N-n) f32.nan) val_inds' MO'
            let breaks = map2 (\m b ->  if (f32.isnan m) || (f32.isnan b)
                                        then 0.0 else (f32.abs m) - b
                                        --used to be nan instead of 0.0
                              ) MO'' BOUND
            in  breaks
        )
  in breakss

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

entry main [m][N] (k: i32) (n: i32) (f: f32)
                  (hfrac: f32) (lam: f32)
                  (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  let (Nss, y_errors, val_indss) = bfast_5 Y y_preds
  let (nss, sigmas) = bfast_6 Y y_errors n k2p2
  let h = t32 ( (r32 n) * hfrac )
  let MO_fsts = bfast_7a y_errors nss h
  let BOUND = bfast_7b N n lam
  let breakss = bfast_8 Nss nss sigmas MO_fsts y_errors val_indss BOUND h n
  in breakss

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- For generating test inputs and outputs

entry bfast_inputs [m][N] (k: i32) (n: i32) (f: f32)
                   (hfrac: f32) (lam: f32)
                   (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  let (Nss, y_errors, val_indss) = bfast_5 Y y_preds
  let (nss, sigmas) = bfast_6 Y y_errors n k2p2
  let h = t32 ( (r32 n) * hfrac )
  let MO_fsts = bfast_7a y_errors nss h
  let BOUND = bfast_7b N n lam
  in (k, n, f, hfrac, lam, Y, X, Xsqr, Xinv, beta0, beta, y_preds, Nss,
      y_errors,  val_indss, nss, sigmas, MO_fsts, BOUND)

entry bfast_1_out  [m][N] (k: i32) (_: i32) (f: f32)
                          (_: f32) (_: f32)
                          (_ : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  in X

entry bfast_2_out  [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  in Xsqr

entry bfast_3_out  [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  in Xinv

entry bfast_4a_out [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let beta0 = bfast_4a X Y n
  in beta0

entry bfast_4b_out [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  in beta

entry bfast_4c_out [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  in y_preds

entry bfast_5_out  [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  let (Nss, y_errors, val_indss) = bfast_5 Y y_preds
  in (Nss, y_errors, val_indss)

entry bfast_6_out  [m][N] (k: i32) (n: i32) (f: f32)
                          (_: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  let (_, y_errors, _) = bfast_5 Y y_preds
  let (nss, sigmas) = bfast_6 Y y_errors n k2p2
  in (nss, sigmas)

entry bfast_7a_out [m][N] (k: i32) (n: i32) (f: f32)
                          (hfrac: f32) (_: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  let (_, y_errors, _) = bfast_5 Y y_preds
  let (nss, _) = bfast_6 Y y_errors n k2p2
  let h = t32 ( (r32 n) * hfrac )
  let MO_fsts = bfast_7a y_errors nss h
  in MO_fsts

entry bfast_7b_out [m][N] (_: i32) (n: i32) (_: f32)
                          (_: f32) (lam: f32)
                          (_ : [m][N]f32) =
  let BOUND = bfast_7b N n lam
  in BOUND

entry bfast_8_out  [m][N] (k: i32) (n: i32) (f: f32)
                          (hfrac: f32) (lam: f32)
                          (Y : [m][N]f32) =
  let k2p2 = 2 * k + 2
  let X = bfast_1 k2p2 N f
  let Xt = transpose X
  let Xsqr = bfast_2 X Xt Y n
  let Xinv = bfast_3 Xsqr
  let beta0 = bfast_4a X Y n
  let beta = bfast_4b Xinv beta0
  let y_preds = bfast_4c Xt beta
  let (Nss, y_errors, val_indss) = bfast_5 Y y_preds
  let (nss, sigmas) = bfast_6 Y y_errors n k2p2
  let h = t32 ( (r32 n) * hfrac )
  let MO_fsts = bfast_7a y_errors nss h
  let BOUND = bfast_7b N n lam
  let breakss = bfast_8 Nss nss sigmas MO_fsts y_errors val_indss BOUND h n
  in breakss


