-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/sahara.in.gz
-- output @ data/sahara.out.gz

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
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
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

----------------------------------------------------
----------------------------------------------------

-- | The core of the alg: the computation for a time series
--   for one pixel.
let bfast [N] (f: f32) (k: i32) (n: i32)
              (hfrac: f32) (lam: f32)
              (y: [N]f32) :
              []f32 =   -- array of length N-n
  ---------------------------------------------------
  -- INVARIANT COMPUTATION, should be hoisted out! --
  ---------------------------------------------------
  let X = mkX (2*k+2) N f
  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| (N*N + 2*N + 1) / (N + 1) - N - 1
  let Xt  = map (map (+zero)) (copy (transpose X))
            |> intrinsics.opaque

  let yh    = unsafe (y[:n])
  let Xh    = unsafe (X[:,:n])
  let Xth   = unsafe (Xt[:n,:])

  let h = t32 ( (r32 n) * hfrac )

  let BOUND = map (\q -> let t   = n+1+q
                         let tmp = logplus ((r32 t) / (r32 n))
                         in  lam * (f32.sqrt tmp)
                  ) (0 ... N-n-1)

  ----------------------------------
  -- 2. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = matmul_filt Xh Xth yh        -- [2k+2][2k+2]

  ----------------------------------
  -- 3. matrix inversion          --
  ----------------------------------
  let Xinv = mat_inv    Xsqr              -- [2k+2][2k+2]

  ----------------------------------------------
  -- 4. several matrix-vector multiplications --
  --    each distributed by itself because    --
  --    sizes do not match                    --
  --------------------------------------------=-
  let beta0= matvecmul_row_filt Xh yh     -- [2k+2]
  let beta = matvecmul_row Xinv beta0     -- [2k+2]
  let y_pred = matvecmul_row Xt beta      -- [N]
               |> intrinsics.opaque
  -- PERFORMANCE BUG: I need to forbid fusion here, otherwise
  --   the innermost `redomap` of computing y_pred will be fused
  --   in the following `scattermap` and will significantly 
  --   degrade performance (probably due to very expensive 
  --   transpositions required for that step.)

  ------------------------------------------------
  -- 5. the following are distributed together  --
  --    inner size is [N]                       --
  --    I hope the concatenation of the results --
  --    of partition does not create existential--
  --    size.
  ------------------------------------------------
  let y_error_all = map2 (\ye yep -> if !(f32.isnan ye) 
                                     then ye-yep else f32.nan
                         ) y y_pred
  let (tups, Ns) = unsafe ( zip2 y_error_all (iota N) |>
                   partitionCos (\(ye,_) -> !(f32.isnan ye)) (0.0, 0) )
  let (y_error, val_inds) = unzip tups
  
  ---------------------------------------------
  -- 6. ns and sigma are distributed together--
  --    size: [n]                            --
  ---------------------------------------------
  let ns = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh |> reduce (+) 0
  let sigma = map (\i -> if i < ns then unsafe y_error[i] else 0.0) (iota n) |>
              map (\ a -> a*a ) |> reduce (+) 0.0
  let sigma = f32.sqrt ( sigma / (r32 (ns-2*k-2)) )

  ------------------------------------
  -- 7. moving sums first: size [h] --
  ------------------------------------
  let MO_fst = map (\i -> unsafe y_error[i + ns-h+1]) (iota h) 
               |> reduce (+) 0.0

  ---------------------------------------------
  -- 8. moving sums computation: size [N-n]  --
  ---------------------------------------------
  let MO = map (\j -> if j >= Ns-ns then 0.0
                      else if j == 0 then MO_fst
                      else  unsafe (-y_error[ns-h+j] + y_error[ns+j])
               ) (0 ... N-n-1) |> scan (+) 0.0

  -- the good values are the first (Ns - ns) elements of MO'
  let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO

  let val_inds' = map (\i ->  if i < Ns - ns 
                              then (unsafe val_inds[i+ns]) - n
                              else -1
                      ) (iota (N-n))
  let full_MO = scatter (replicate (N-n) f32.nan) val_inds' MO'
  
  -- line 10: BOUND computation (will be hoisted)

  let breaks = map2 (\m b -> if (f32.isnan m) || (f32.isnan b)
                             then 0.0 else (f32.abs m) - b --used to be nan instead of 0.0
                    ) full_MO BOUND
  in  breaks

-- | entry point
entry main [m][N] (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (images : [m][N]f32) :
                  ([m][]f32) =
  let res = map (bfast freq k n hfrac lam) images
  in  res
