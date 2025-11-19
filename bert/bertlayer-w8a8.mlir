#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (0, 0)>
#map5 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map7 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map8 = affine_map<(d0, d1) -> (d1, d0)>
#map9 = affine_map<(d0, d1) -> (d0, 0)>
#map10 = affine_map<(d0) -> (d0)>
#map11 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map15 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map16 = affine_map<(d0) -> (0)>
module {
  func.func @subgraph0(%arg0: tensor<1x9x768xf16>, %arg1: tensor<768x768xi8>, %arg2: tensor<768xf16>, %arg3: tensor<768xf16>, %arg4: tensor<768x768xi8>, %arg5: tensor<768xf16>, %arg6: tensor<768xf16>, %arg7: tensor<768x768xi8>, %arg8: tensor<768xf16>, %arg9: tensor<768xf16>, %arg10: tensor<768x768xi8>, %arg11: tensor<768xf16>, %arg12: tensor<768xf16>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xi8>, %arg16: tensor<3072xf16>, %arg17: tensor<3072xf16>, %arg18: tensor<768x3072xi8>, %arg19: tensor<768xf16>, %arg20: tensor<768xf16>, %arg21: tensor<768xf32>, %arg22: tensor<768xf32>) -> tensor<1x9x768xf16> {
    %0 = tensor.empty() : tensor<1x9xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%0 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.minf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %2 = tensor.empty() : tensor<1x9xf16>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%2 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.maxf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %4 = tensor.empty() : tensor<1x9xf16>
    %5 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %cst : tensor<1x9xf16>, tensor<1x9xf16>) outs(%4 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.minf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %6 = tensor.empty() : tensor<1x9xf16>
    %7 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_0 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%6 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %8 = tensor.empty() : tensor<1x9xf16>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x9xf16>) outs(%8 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.negf %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %10 = tensor.empty() : tensor<1x9xf16>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%9, %7 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%10 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_1 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %cst_2 = arith.constant dense<1.270000e+02> : tensor<1x1xf16>
    %12 = tensor.empty() : tensor<1x1xf16>
    %13 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<1x1xf16>) outs(%12 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cst_163 = arith.constant 1.000000e+00 : f16
      %481 = arith.divf %cst_163, %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x1xf16>
    %14 = tensor.empty() : tensor<1x9xf16>
    %15 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%11, %13 : tensor<1x9xf16>, tensor<1x1xf16>) outs(%14 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %16 = tensor.empty() : tensor<1x9xf32>
    %17 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<1x9xf16>) outs(%16 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %18 = tensor.empty() : tensor<1x9xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<1x9xf32>) outs(%18 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 9.99999974E-6 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9xf32>
    %20 = tensor.empty() : tensor<1x9xf16>
    %21 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<1x9xf32>) outs(%20 : tensor<1x9xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %22 = tensor.empty() : tensor<1x9xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<1x9xf16>) outs(%22 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %expanded = tensor.expand_shape %23 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<1x9x1xf16>
    %24 = tensor.empty() : tensor<1x9x1xf32>
    %25 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x9x1xf32>) outs(%24 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<1x1x1xf32>
    %26 = tensor.empty() : tensor<1x9x1xf32>
    %27 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %cst_6 : tensor<1x9x1xf32>, tensor<1x1x1xf32>) outs(%26 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %28 = tensor.empty() : tensor<1x9x768xf32>
    %29 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%28 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %30 = tensor.empty() : tensor<1x9x768xf32>
    %31 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %27 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%30 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %32 = tensor.empty() : tensor<1x9x768xf32>
    %33 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<1x9x768xf32>) outs(%32 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.floor %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %34 = tensor.empty() : tensor<1x9x1xf32>
    %35 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_4 : tensor<1x9x1xf16>) outs(%34 : tensor<1x9x1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %36 = tensor.empty() : tensor<1x9x768xf32>
    %37 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%36 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %38 = tensor.empty() : tensor<1x9x768xf32>
    %39 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37 : tensor<1x9x768xf32>) outs(%38 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant -1.270000e+02 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %40 = tensor.empty() : tensor<1x9x768xf32>
    %41 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39 : tensor<1x9x768xf32>) outs(%40 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 0xFF800000 : f32
      %cst_164 = arith.constant 1.270000e+02 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %42 = tensor.empty() : tensor<1x9x768xi8>
    %43 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41 : tensor<1x9x768xf32>) outs(%42 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %481 = arith.fptosi %in : f32 to i8
      linalg.yield %481 : i8
    } -> tensor<1x9x768xi8>
    %cst_7 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %44 = tensor.empty() : tensor<768x768xi8>
    %45 = linalg.generic {indexing_maps = [#map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<768x768xi8>) outs(%44 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed = tensor.collapse_shape %43 [[0, 1], [2]] : tensor<1x9x768xi8> into tensor<9x768xi8>
    %collapsed_8 = tensor.collapse_shape %23 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_9 = tensor.expand_shape %collapsed_8 [[0, 1]] : tensor<9xf32> into tensor<9x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<9x768xf32>
    %46 = tensor.empty() : tensor<9x768xf32>
    %47 = linalg.generic {indexing_maps = [#map9, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_9, %cst_10 : tensor<9x1xf32>, tensor<9x768xf32>) outs(%46 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %cst_11 = arith.constant dense<0> : tensor<9x768xi32>
    %48 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed, %45 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_11 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %49 = tensor.empty() : tensor<9x768xf32>
    %50 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%48 : tensor<9x768xi32>) outs(%49 : tensor<9x768xf32>) {
    ^bb0(%in: i32, %out: f32):
      %481 = arith.sitofp %in : i32 to f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %51 = tensor.empty() : tensor<9x768xf32>
    %52 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%50, %47 : tensor<9x768xf32>, tensor<9x768xf32>) outs(%51 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %53 = tensor.empty() : tensor<768xf32>
    %54 = linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel"]} ins(%arg2 : tensor<768xf16>) outs(%53 : tensor<768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<768xf32>
    %expanded_12 = tensor.expand_shape %54 [[0, 1]] : tensor<768xf32> into tensor<1x768xf32>
    %55 = tensor.empty() : tensor<9x768xf32>
    %56 = linalg.generic {indexing_maps = [#map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%52, %expanded_12 : tensor<9x768xf32>, tensor<1x768xf32>) outs(%55 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %expanded_13 = tensor.expand_shape %56 [[0, 1], [2]] : tensor<9x768xf32> into tensor<1x9x768xf32>
    %57 = tensor.empty() : tensor<1x9x768xf16>
    %58 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<1x9x768xf32>) outs(%57 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %expanded_14 = tensor.expand_shape %arg3 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %59 = tensor.empty() : tensor<1x9x768xf16>
    %60 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%58, %expanded_14 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%59 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %61 = tensor.empty() : tensor<1x9xf16>
    %62 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%61 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.minf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %63 = tensor.empty() : tensor<1x9xf16>
    %64 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%63 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.maxf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %65 = tensor.empty() : tensor<1x9xf16>
    %66 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%62, %cst_15 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%65 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.minf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %67 = tensor.empty() : tensor<1x9xf16>
    %68 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%64, %cst_16 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%67 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %69 = tensor.empty() : tensor<1x9xf16>
    %70 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%66 : tensor<1x9xf16>) outs(%69 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.negf %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %71 = tensor.empty() : tensor<1x9xf16>
    %72 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%70, %68 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%71 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_17 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %cst_18 = arith.constant dense<1.270000e+02> : tensor<1x1xf16>
    %73 = tensor.empty() : tensor<1x1xf16>
    %74 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_18 : tensor<1x1xf16>) outs(%73 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cst_163 = arith.constant 1.000000e+00 : f16
      %481 = arith.divf %cst_163, %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x1xf16>
    %75 = tensor.empty() : tensor<1x9xf16>
    %76 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%72, %74 : tensor<1x9xf16>, tensor<1x1xf16>) outs(%75 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_19 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %77 = tensor.empty() : tensor<1x9xf32>
    %78 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%76 : tensor<1x9xf16>) outs(%77 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %79 = tensor.empty() : tensor<1x9xf32>
    %80 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%78 : tensor<1x9xf32>) outs(%79 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 9.99999974E-6 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9xf32>
    %81 = tensor.empty() : tensor<1x9xf16>
    %82 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%80 : tensor<1x9xf32>) outs(%81 : tensor<1x9xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %83 = tensor.empty() : tensor<1x9xf32>
    %84 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%82 : tensor<1x9xf16>) outs(%83 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %expanded_20 = tensor.expand_shape %84 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_21 = arith.constant dense<0.000000e+00> : tensor<1x9x1xf16>
    %85 = tensor.empty() : tensor<1x9x1xf32>
    %86 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_20 : tensor<1x9x1xf32>) outs(%85 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_22 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_23 = arith.constant dense<1.000000e+00> : tensor<1x1x1xf32>
    %87 = tensor.empty() : tensor<1x9x1xf32>
    %88 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %cst_23 : tensor<1x9x1xf32>, tensor<1x1x1xf32>) outs(%87 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %89 = tensor.empty() : tensor<1x9x768xf32>
    %90 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%89 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %91 = tensor.empty() : tensor<1x9x768xf32>
    %92 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %88 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%91 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %93 = tensor.empty() : tensor<1x9x768xf32>
    %94 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92 : tensor<1x9x768xf32>) outs(%93 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.floor %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %95 = tensor.empty() : tensor<1x9x1xf32>
    %96 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_21 : tensor<1x9x1xf16>) outs(%95 : tensor<1x9x1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %97 = tensor.empty() : tensor<1x9x768xf32>
    %98 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %96 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%97 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %99 = tensor.empty() : tensor<1x9x768xf32>
    %100 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%98 : tensor<1x9x768xf32>) outs(%99 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant -1.270000e+02 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %101 = tensor.empty() : tensor<1x9x768xf32>
    %102 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100 : tensor<1x9x768xf32>) outs(%101 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 0xFF800000 : f32
      %cst_164 = arith.constant 1.270000e+02 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %103 = tensor.empty() : tensor<1x9x768xi8>
    %104 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x9x768xf32>) outs(%103 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %481 = arith.fptosi %in : f32 to i8
      linalg.yield %481 : i8
    } -> tensor<1x9x768xi8>
    %cst_24 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %105 = tensor.empty() : tensor<768x768xi8>
    %106 = linalg.generic {indexing_maps = [#map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<768x768xi8>) outs(%105 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_25 = tensor.collapse_shape %104 [[0, 1], [2]] : tensor<1x9x768xi8> into tensor<9x768xi8>
    %collapsed_26 = tensor.collapse_shape %84 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_27 = tensor.expand_shape %collapsed_26 [[0, 1]] : tensor<9xf32> into tensor<9x1xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<9x768xf32>
    %107 = tensor.empty() : tensor<9x768xf32>
    %108 = linalg.generic {indexing_maps = [#map9, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_27, %cst_28 : tensor<9x1xf32>, tensor<9x768xf32>) outs(%107 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %cst_29 = arith.constant dense<0> : tensor<9x768xi32>
    %109 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_25, %106 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_29 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %110 = tensor.empty() : tensor<9x768xf32>
    %111 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%109 : tensor<9x768xi32>) outs(%110 : tensor<9x768xf32>) {
    ^bb0(%in: i32, %out: f32):
      %481 = arith.sitofp %in : i32 to f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %112 = tensor.empty() : tensor<9x768xf32>
    %113 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%111, %108 : tensor<9x768xf32>, tensor<9x768xf32>) outs(%112 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %114 = tensor.empty() : tensor<768xf32>
    %115 = linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel"]} ins(%arg5 : tensor<768xf16>) outs(%114 : tensor<768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<768xf32>
    %expanded_30 = tensor.expand_shape %115 [[0, 1]] : tensor<768xf32> into tensor<1x768xf32>
    %116 = tensor.empty() : tensor<9x768xf32>
    %117 = linalg.generic {indexing_maps = [#map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%113, %expanded_30 : tensor<9x768xf32>, tensor<1x768xf32>) outs(%116 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %expanded_31 = tensor.expand_shape %117 [[0, 1], [2]] : tensor<9x768xf32> into tensor<1x9x768xf32>
    %118 = tensor.empty() : tensor<1x9x768xf16>
    %119 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_31 : tensor<1x9x768xf32>) outs(%118 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %expanded_32 = tensor.expand_shape %arg6 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %120 = tensor.empty() : tensor<1x9x768xf16>
    %121 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %expanded_32 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%120 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %122 = tensor.empty() : tensor<1x9xf16>
    %123 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%122 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.minf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %124 = tensor.empty() : tensor<1x9xf16>
    %125 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%124 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.maxf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_33 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %126 = tensor.empty() : tensor<1x9xf16>
    %127 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%123, %cst_33 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%126 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.minf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_34 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %128 = tensor.empty() : tensor<1x9xf16>
    %129 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%125, %cst_34 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%128 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %130 = tensor.empty() : tensor<1x9xf16>
    %131 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%127 : tensor<1x9xf16>) outs(%130 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.negf %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %132 = tensor.empty() : tensor<1x9xf16>
    %133 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%131, %129 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%132 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_35 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %cst_36 = arith.constant dense<1.270000e+02> : tensor<1x1xf16>
    %134 = tensor.empty() : tensor<1x1xf16>
    %135 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_36 : tensor<1x1xf16>) outs(%134 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cst_163 = arith.constant 1.000000e+00 : f16
      %481 = arith.divf %cst_163, %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x1xf16>
    %136 = tensor.empty() : tensor<1x9xf16>
    %137 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%133, %135 : tensor<1x9xf16>, tensor<1x1xf16>) outs(%136 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_37 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %138 = tensor.empty() : tensor<1x9xf32>
    %139 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%137 : tensor<1x9xf16>) outs(%138 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %140 = tensor.empty() : tensor<1x9xf32>
    %141 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%139 : tensor<1x9xf32>) outs(%140 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 9.99999974E-6 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9xf32>
    %142 = tensor.empty() : tensor<1x9xf16>
    %143 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%141 : tensor<1x9xf32>) outs(%142 : tensor<1x9xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %144 = tensor.empty() : tensor<1x9xf32>
    %145 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%143 : tensor<1x9xf16>) outs(%144 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %expanded_38 = tensor.expand_shape %145 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_39 = arith.constant dense<0.000000e+00> : tensor<1x9x1xf16>
    %146 = tensor.empty() : tensor<1x9x1xf32>
    %147 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_38 : tensor<1x9x1xf32>) outs(%146 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_40 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_41 = arith.constant dense<1.000000e+00> : tensor<1x1x1xf32>
    %148 = tensor.empty() : tensor<1x9x1xf32>
    %149 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%147, %cst_41 : tensor<1x9x1xf32>, tensor<1x1x1xf32>) outs(%148 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %150 = tensor.empty() : tensor<1x9x768xf32>
    %151 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x9x768xf16>) outs(%150 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %152 = tensor.empty() : tensor<1x9x768xf32>
    %153 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%151, %149 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%152 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %154 = tensor.empty() : tensor<1x9x768xf32>
    %155 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%153 : tensor<1x9x768xf32>) outs(%154 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.floor %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %156 = tensor.empty() : tensor<1x9x1xf32>
    %157 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_39 : tensor<1x9x1xf16>) outs(%156 : tensor<1x9x1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %158 = tensor.empty() : tensor<1x9x768xf32>
    %159 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%155, %157 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%158 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %160 = tensor.empty() : tensor<1x9x768xf32>
    %161 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%159 : tensor<1x9x768xf32>) outs(%160 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant -1.270000e+02 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %162 = tensor.empty() : tensor<1x9x768xf32>
    %163 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161 : tensor<1x9x768xf32>) outs(%162 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 0xFF800000 : f32
      %cst_164 = arith.constant 1.270000e+02 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %164 = tensor.empty() : tensor<1x9x768xi8>
    %165 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%163 : tensor<1x9x768xf32>) outs(%164 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %481 = arith.fptosi %in : f32 to i8
      linalg.yield %481 : i8
    } -> tensor<1x9x768xi8>
    %cst_42 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %166 = tensor.empty() : tensor<768x768xi8>
    %167 = linalg.generic {indexing_maps = [#map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7 : tensor<768x768xi8>) outs(%166 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_43 = tensor.collapse_shape %165 [[0, 1], [2]] : tensor<1x9x768xi8> into tensor<9x768xi8>
    %collapsed_44 = tensor.collapse_shape %145 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_45 = tensor.expand_shape %collapsed_44 [[0, 1]] : tensor<9xf32> into tensor<9x1xf32>
    %cst_46 = arith.constant dense<0.000000e+00> : tensor<9x768xf32>
    %168 = tensor.empty() : tensor<9x768xf32>
    %169 = linalg.generic {indexing_maps = [#map9, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_45, %cst_46 : tensor<9x1xf32>, tensor<9x768xf32>) outs(%168 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %cst_47 = arith.constant dense<0> : tensor<9x768xi32>
    %170 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_43, %167 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_47 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %171 = tensor.empty() : tensor<9x768xf32>
    %172 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%170 : tensor<9x768xi32>) outs(%171 : tensor<9x768xf32>) {
    ^bb0(%in: i32, %out: f32):
      %481 = arith.sitofp %in : i32 to f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %173 = tensor.empty() : tensor<9x768xf32>
    %174 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%172, %169 : tensor<9x768xf32>, tensor<9x768xf32>) outs(%173 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %175 = tensor.empty() : tensor<768xf32>
    %176 = linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel"]} ins(%arg8 : tensor<768xf16>) outs(%175 : tensor<768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<768xf32>
    %expanded_48 = tensor.expand_shape %176 [[0, 1]] : tensor<768xf32> into tensor<1x768xf32>
    %177 = tensor.empty() : tensor<9x768xf32>
    %178 = linalg.generic {indexing_maps = [#map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%174, %expanded_48 : tensor<9x768xf32>, tensor<1x768xf32>) outs(%177 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %expanded_49 = tensor.expand_shape %178 [[0, 1], [2]] : tensor<9x768xf32> into tensor<1x9x768xf32>
    %179 = tensor.empty() : tensor<1x9x768xf16>
    %180 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_49 : tensor<1x9x768xf32>) outs(%179 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %expanded_50 = tensor.expand_shape %arg9 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %181 = tensor.empty() : tensor<1x9x768xf16>
    %182 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180, %expanded_50 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%181 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %expanded_51 = tensor.expand_shape %60 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_52 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %183 = tensor.empty() : tensor<1x12x9x64xf16>
    %184 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_51 : tensor<1x9x12x64xf16>) outs(%183 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %expanded_53 = tensor.expand_shape %121 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_54 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %185 = tensor.empty() : tensor<1x12x9x64xf16>
    %186 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_53 : tensor<1x9x12x64xf16>) outs(%185 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %expanded_55 = tensor.expand_shape %182 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_56 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %187 = tensor.empty() : tensor<1x12x9x64xf16>
    %188 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_55 : tensor<1x9x12x64xf16>) outs(%187 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %cst_57 = arith.constant 0.000000e+00 : f16
    %cst_58 = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
    %cst_59 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %189 = tensor.empty() : tensor<1x12x64x9xf16>
    %190 = linalg.generic {indexing_maps = [#map14, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186 : tensor<1x12x9x64xf16>) outs(%189 : tensor<1x12x64x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x64x9xf16>
    %collapsed_60 = tensor.collapse_shape %184 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %collapsed_61 = tensor.collapse_shape %190 [[0, 1], [2], [3]] : tensor<1x12x64x9xf16> into tensor<12x64x9xf16>
    %cst_62 = arith.constant 0.000000e+00 : f16
    %191 = tensor.empty() : tensor<12x9x9xf16>
    %192 = linalg.fill ins(%cst_62 : f16) outs(%191 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %193 = linalg.batch_matmul ins(%collapsed_60, %collapsed_61 : tensor<12x9x64xf16>, tensor<12x64x9xf16>) outs(%192 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %cst_63 = arith.constant 1.250000e-01 : f16
    %cst_64 = arith.constant dense<1.250000e-01> : tensor<12x9x9xf16>
    %194 = tensor.empty() : tensor<12x9x9xf16>
    %195 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%193, %cst_64 : tensor<12x9x9xf16>, tensor<12x9x9xf16>) outs(%194 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<12x9x9xf16>
    %expanded_65 = tensor.expand_shape %cst_58 [[0, 1], [2]] : tensor<9x9xf16> into tensor<1x9x9xf16>
    %196 = tensor.empty() : tensor<12x9x9xf16>
    %197 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195, %expanded_65 : tensor<12x9x9xf16>, tensor<1x9x9xf16>) outs(%196 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<12x9x9xf16>
    %198 = tensor.empty() : tensor<12x9xf16>
    %cst_66 = arith.constant -6.550400e+04 : f16
    %199 = linalg.fill ins(%cst_66 : f16) outs(%198 : tensor<12x9xf16>) -> tensor<12x9xf16>
    %reduced = linalg.reduce ins(%197 : tensor<12x9x9xf16>) outs(%199 : tensor<12x9xf16>) dimensions = [2] 
      (%in: f16, %init: f16) {
        %481 = arith.maxf %in, %init : f16
        linalg.yield %481 : f16
      }
    %expanded_67 = tensor.expand_shape %reduced [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %200 = tensor.empty() : tensor<12x9x9xf16>
    %201 = linalg.generic {indexing_maps = [#map, #map15, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%197, %expanded_67 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%200 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.subf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<12x9x9xf16>
    %202 = math.exp %201 : tensor<12x9x9xf16>
    %203 = tensor.empty() : tensor<12x9xf16>
    %cst_68 = arith.constant 0.000000e+00 : f16
    %204 = linalg.fill ins(%cst_68 : f16) outs(%203 : tensor<12x9xf16>) -> tensor<12x9xf16>
    %reduced_69 = linalg.reduce ins(%202 : tensor<12x9x9xf16>) outs(%204 : tensor<12x9xf16>) dimensions = [2] 
      (%in: f16, %init: f16) {
        %481 = arith.addf %in, %init : f16
        linalg.yield %481 : f16
      }
    %expanded_70 = tensor.expand_shape %reduced_69 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %205 = tensor.empty() : tensor<12x9x1xf16>
    %206 = linalg.generic {indexing_maps = [#map15, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_70 : tensor<12x9x1xf16>) outs(%205 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = math.log %in : f16
      linalg.yield %481 : f16
    } -> tensor<12x9x1xf16>
    %207 = tensor.empty() : tensor<12x9x1xf16>
    %208 = linalg.generic {indexing_maps = [#map15, #map15, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_67, %206 : tensor<12x9x1xf16>, tensor<12x9x1xf16>) outs(%207 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<12x9x1xf16>
    %209 = tensor.empty() : tensor<12x9x9xf16>
    %210 = linalg.generic {indexing_maps = [#map, #map15, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%197, %208 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%209 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.subf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<12x9x9xf16>
    %211 = math.exp %210 : tensor<12x9x9xf16>
    %collapsed_71 = tensor.collapse_shape %208 [[0], [1, 2]] : tensor<12x9x1xf16> into tensor<12x9xf16>
    %expanded_72 = tensor.expand_shape %collapsed_71 [[0, 1], [2]] : tensor<12x9xf16> into tensor<1x12x9xf16>
    %collapsed_73 = tensor.collapse_shape %188 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %cst_74 = arith.constant 0.000000e+00 : f16
    %212 = tensor.empty() : tensor<12x9x64xf16>
    %213 = linalg.fill ins(%cst_74 : f16) outs(%212 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %214 = linalg.batch_matmul ins(%211, %collapsed_73 : tensor<12x9x9xf16>, tensor<12x9x64xf16>) outs(%213 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %expanded_75 = tensor.expand_shape %214 [[0, 1], [2], [3]] : tensor<12x9x64xf16> into tensor<1x12x9x64xf16>
    %cst_76 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %215 = tensor.empty() : tensor<1x9x12x64xf16>
    %216 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_75 : tensor<1x12x9x64xf16>) outs(%215 : tensor<1x9x12x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x9x12x64xf16>
    %collapsed_77 = tensor.collapse_shape %216 [[0], [1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<1x9x768xf16>
    %217 = tensor.empty() : tensor<1x9xf16>
    %218 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_77 : tensor<1x9x768xf16>) outs(%217 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.minf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %219 = tensor.empty() : tensor<1x9xf16>
    %220 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_77 : tensor<1x9x768xf16>) outs(%219 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.maxf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_78 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %221 = tensor.empty() : tensor<1x9xf16>
    %222 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%218, %cst_78 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%221 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.minf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_79 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %223 = tensor.empty() : tensor<1x9xf16>
    %224 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%220, %cst_79 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%223 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %225 = tensor.empty() : tensor<1x9xf16>
    %226 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%222 : tensor<1x9xf16>) outs(%225 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.negf %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %227 = tensor.empty() : tensor<1x9xf16>
    %228 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%226, %224 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%227 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_80 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %cst_81 = arith.constant dense<1.270000e+02> : tensor<1x1xf16>
    %229 = tensor.empty() : tensor<1x1xf16>
    %230 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_81 : tensor<1x1xf16>) outs(%229 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cst_163 = arith.constant 1.000000e+00 : f16
      %481 = arith.divf %cst_163, %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x1xf16>
    %231 = tensor.empty() : tensor<1x9xf16>
    %232 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%228, %230 : tensor<1x9xf16>, tensor<1x1xf16>) outs(%231 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_82 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %233 = tensor.empty() : tensor<1x9xf32>
    %234 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%232 : tensor<1x9xf16>) outs(%233 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %235 = tensor.empty() : tensor<1x9xf32>
    %236 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%234 : tensor<1x9xf32>) outs(%235 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 9.99999974E-6 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9xf32>
    %237 = tensor.empty() : tensor<1x9xf16>
    %238 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%236 : tensor<1x9xf32>) outs(%237 : tensor<1x9xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %239 = tensor.empty() : tensor<1x9xf32>
    %240 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%238 : tensor<1x9xf16>) outs(%239 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %expanded_83 = tensor.expand_shape %240 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_84 = arith.constant dense<0.000000e+00> : tensor<1x9x1xf16>
    %241 = tensor.empty() : tensor<1x9x1xf32>
    %242 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_83 : tensor<1x9x1xf32>) outs(%241 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_85 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_86 = arith.constant dense<1.000000e+00> : tensor<1x1x1xf32>
    %243 = tensor.empty() : tensor<1x9x1xf32>
    %244 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%242, %cst_86 : tensor<1x9x1xf32>, tensor<1x1x1xf32>) outs(%243 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %245 = tensor.empty() : tensor<1x9x768xf32>
    %246 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_77 : tensor<1x9x768xf16>) outs(%245 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %247 = tensor.empty() : tensor<1x9x768xf32>
    %248 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%246, %244 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%247 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %249 = tensor.empty() : tensor<1x9x768xf32>
    %250 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%248 : tensor<1x9x768xf32>) outs(%249 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.floor %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %251 = tensor.empty() : tensor<1x9x1xf32>
    %252 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_84 : tensor<1x9x1xf16>) outs(%251 : tensor<1x9x1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %253 = tensor.empty() : tensor<1x9x768xf32>
    %254 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250, %252 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%253 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %255 = tensor.empty() : tensor<1x9x768xf32>
    %256 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254 : tensor<1x9x768xf32>) outs(%255 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant -1.270000e+02 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %257 = tensor.empty() : tensor<1x9x768xf32>
    %258 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%256 : tensor<1x9x768xf32>) outs(%257 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 0xFF800000 : f32
      %cst_164 = arith.constant 1.270000e+02 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %259 = tensor.empty() : tensor<1x9x768xi8>
    %260 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%258 : tensor<1x9x768xf32>) outs(%259 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %481 = arith.fptosi %in : f32 to i8
      linalg.yield %481 : i8
    } -> tensor<1x9x768xi8>
    %cst_87 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %261 = tensor.empty() : tensor<768x768xi8>
    %262 = linalg.generic {indexing_maps = [#map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg10 : tensor<768x768xi8>) outs(%261 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_88 = tensor.collapse_shape %260 [[0, 1], [2]] : tensor<1x9x768xi8> into tensor<9x768xi8>
    %collapsed_89 = tensor.collapse_shape %240 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_90 = tensor.expand_shape %collapsed_89 [[0, 1]] : tensor<9xf32> into tensor<9x1xf32>
    %cst_91 = arith.constant dense<0.000000e+00> : tensor<9x768xf32>
    %263 = tensor.empty() : tensor<9x768xf32>
    %264 = linalg.generic {indexing_maps = [#map9, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_90, %cst_91 : tensor<9x1xf32>, tensor<9x768xf32>) outs(%263 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %cst_92 = arith.constant dense<0> : tensor<9x768xi32>
    %265 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_88, %262 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_92 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %266 = tensor.empty() : tensor<9x768xf32>
    %267 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%265 : tensor<9x768xi32>) outs(%266 : tensor<9x768xf32>) {
    ^bb0(%in: i32, %out: f32):
      %481 = arith.sitofp %in : i32 to f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %268 = tensor.empty() : tensor<9x768xf32>
    %269 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%267, %264 : tensor<9x768xf32>, tensor<9x768xf32>) outs(%268 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %270 = tensor.empty() : tensor<768xf32>
    %271 = linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel"]} ins(%arg11 : tensor<768xf16>) outs(%270 : tensor<768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<768xf32>
    %expanded_93 = tensor.expand_shape %271 [[0, 1]] : tensor<768xf32> into tensor<1x768xf32>
    %272 = tensor.empty() : tensor<9x768xf32>
    %273 = linalg.generic {indexing_maps = [#map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%269, %expanded_93 : tensor<9x768xf32>, tensor<1x768xf32>) outs(%272 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %expanded_94 = tensor.expand_shape %273 [[0, 1], [2]] : tensor<9x768xf32> into tensor<1x9x768xf32>
    %274 = tensor.empty() : tensor<1x9x768xf16>
    %275 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_94 : tensor<1x9x768xf32>) outs(%274 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %expanded_95 = tensor.expand_shape %arg12 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %276 = tensor.empty() : tensor<1x9x768xf16>
    %277 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275, %expanded_95 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%276 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %278 = tensor.empty() : tensor<1x9x768xf16>
    %279 = linalg.generic {indexing_maps = [#map7, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %277 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%278 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %280 = tensor.empty() : tensor<1x9x768xf32>
    %281 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%279 : tensor<1x9x768xf16>) outs(%280 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %282 = tensor.empty() : tensor<1x9xf32>
    %cst_96 = arith.constant 0.000000e+00 : f32
    %283 = linalg.fill ins(%cst_96 : f32) outs(%282 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %reduced_97 = linalg.reduce ins(%281 : tensor<1x9x768xf32>) outs(%283 : tensor<1x9xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %481 = arith.addf %in, %init : f32
        linalg.yield %481 : f32
      }
    %expanded_98 = tensor.expand_shape %reduced_97 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_99 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %284 = tensor.empty() : tensor<1xf32>
    %285 = linalg.generic {indexing_maps = [#map16, #map10], iterator_types = ["parallel"]} ins(%cst_99 : tensor<1xf32>) outs(%284 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1xf32>
    %expanded_100 = tensor.expand_shape %285 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %286 = tensor.empty() : tensor<1x9x1xf32>
    %287 = linalg.generic {indexing_maps = [#map6, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_100, %expanded_98 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%286 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %288 = tensor.empty() : tensor<1x9x768xf32>
    %289 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%281, %287 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%288 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.subf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %290 = tensor.empty() : tensor<1x9x768xf32>
    %291 = linalg.generic {indexing_maps = [#map7, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%289, %289 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%290 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %292 = tensor.empty() : tensor<1x9xf32>
    %cst_101 = arith.constant 0.000000e+00 : f32
    %293 = linalg.fill ins(%cst_101 : f32) outs(%292 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %reduced_102 = linalg.reduce ins(%291 : tensor<1x9x768xf32>) outs(%293 : tensor<1x9xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %481 = arith.addf %in, %init : f32
        linalg.yield %481 : f32
      }
    %expanded_103 = tensor.expand_shape %reduced_102 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_104 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %294 = tensor.empty() : tensor<1xf32>
    %295 = linalg.generic {indexing_maps = [#map16, #map10], iterator_types = ["parallel"]} ins(%cst_104 : tensor<1xf32>) outs(%294 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1xf32>
    %expanded_105 = tensor.expand_shape %295 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %296 = tensor.empty() : tensor<1x9x1xf32>
    %297 = linalg.generic {indexing_maps = [#map6, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_105, %expanded_103 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%296 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_106 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %298 = tensor.empty() : tensor<1x9x1xf32>
    %299 = linalg.generic {indexing_maps = [#map5, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%297, %cst_106 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%298 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %300 = tensor.empty() : tensor<1x9x1xf32>
    %301 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%299 : tensor<1x9x1xf32>) outs(%300 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.rsqrt %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %302 = tensor.empty() : tensor<1x9x768xf32>
    %303 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%281, %287 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%302 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.subf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %304 = tensor.empty() : tensor<1x9x768xf32>
    %305 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%303, %301 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%304 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %expanded_107 = tensor.expand_shape %arg13 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %306 = tensor.empty() : tensor<1x9x768xf32>
    %307 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%305, %expanded_107 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%306 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %expanded_108 = tensor.expand_shape %arg14 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %308 = tensor.empty() : tensor<1x9x768xf32>
    %309 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%307, %expanded_108 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%308 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %310 = tensor.empty() : tensor<1x9x768xf16>
    %311 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%309 : tensor<1x9x768xf32>) outs(%310 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %312 = tensor.empty() : tensor<1x9xf16>
    %313 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%311 : tensor<1x9x768xf16>) outs(%312 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.minf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %314 = tensor.empty() : tensor<1x9xf16>
    %315 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%311 : tensor<1x9x768xf16>) outs(%314 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.maxf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_109 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %316 = tensor.empty() : tensor<1x9xf16>
    %317 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%313, %cst_109 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%316 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.minf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_110 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %318 = tensor.empty() : tensor<1x9xf16>
    %319 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%315, %cst_110 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%318 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %320 = tensor.empty() : tensor<1x9xf16>
    %321 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%317 : tensor<1x9xf16>) outs(%320 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.negf %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %322 = tensor.empty() : tensor<1x9xf16>
    %323 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%321, %319 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%322 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_111 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %cst_112 = arith.constant dense<1.270000e+02> : tensor<1x1xf16>
    %324 = tensor.empty() : tensor<1x1xf16>
    %325 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_112 : tensor<1x1xf16>) outs(%324 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cst_163 = arith.constant 1.000000e+00 : f16
      %481 = arith.divf %cst_163, %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x1xf16>
    %326 = tensor.empty() : tensor<1x9xf16>
    %327 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%323, %325 : tensor<1x9xf16>, tensor<1x1xf16>) outs(%326 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_113 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %328 = tensor.empty() : tensor<1x9xf32>
    %329 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%327 : tensor<1x9xf16>) outs(%328 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %330 = tensor.empty() : tensor<1x9xf32>
    %331 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%329 : tensor<1x9xf32>) outs(%330 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 9.99999974E-6 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9xf32>
    %332 = tensor.empty() : tensor<1x9xf16>
    %333 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%331 : tensor<1x9xf32>) outs(%332 : tensor<1x9xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %334 = tensor.empty() : tensor<1x9xf32>
    %335 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%333 : tensor<1x9xf16>) outs(%334 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %expanded_114 = tensor.expand_shape %335 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_115 = arith.constant dense<0.000000e+00> : tensor<1x9x1xf16>
    %336 = tensor.empty() : tensor<1x9x1xf32>
    %337 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_114 : tensor<1x9x1xf32>) outs(%336 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_116 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_117 = arith.constant dense<1.000000e+00> : tensor<1x1x1xf32>
    %338 = tensor.empty() : tensor<1x9x1xf32>
    %339 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337, %cst_117 : tensor<1x9x1xf32>, tensor<1x1x1xf32>) outs(%338 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %340 = tensor.empty() : tensor<1x9x768xf32>
    %341 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%311 : tensor<1x9x768xf16>) outs(%340 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %342 = tensor.empty() : tensor<1x9x768xf32>
    %343 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%341, %339 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%342 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %344 = tensor.empty() : tensor<1x9x768xf32>
    %345 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%343 : tensor<1x9x768xf32>) outs(%344 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.floor %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %346 = tensor.empty() : tensor<1x9x1xf32>
    %347 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_115 : tensor<1x9x1xf16>) outs(%346 : tensor<1x9x1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %348 = tensor.empty() : tensor<1x9x768xf32>
    %349 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%345, %347 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%348 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %350 = tensor.empty() : tensor<1x9x768xf32>
    %351 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%349 : tensor<1x9x768xf32>) outs(%350 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant -1.270000e+02 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %352 = tensor.empty() : tensor<1x9x768xf32>
    %353 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%351 : tensor<1x9x768xf32>) outs(%352 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 0xFF800000 : f32
      %cst_164 = arith.constant 1.270000e+02 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x768xf32>
    %354 = tensor.empty() : tensor<1x9x768xi8>
    %355 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%353 : tensor<1x9x768xf32>) outs(%354 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %481 = arith.fptosi %in : f32 to i8
      linalg.yield %481 : i8
    } -> tensor<1x9x768xi8>
    %cst_118 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %356 = tensor.empty() : tensor<768x3072xi8>
    %357 = linalg.generic {indexing_maps = [#map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg15 : tensor<3072x768xi8>) outs(%356 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x3072xi8>
    %collapsed_119 = tensor.collapse_shape %355 [[0, 1], [2]] : tensor<1x9x768xi8> into tensor<9x768xi8>
    %collapsed_120 = tensor.collapse_shape %335 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_121 = tensor.expand_shape %collapsed_120 [[0, 1]] : tensor<9xf32> into tensor<9x1xf32>
    %cst_122 = arith.constant dense<0.000000e+00> : tensor<9x3072xf32>
    %358 = tensor.empty() : tensor<9x3072xf32>
    %359 = linalg.generic {indexing_maps = [#map9, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_121, %cst_122 : tensor<9x1xf32>, tensor<9x3072xf32>) outs(%358 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x3072xf32>
    %cst_123 = arith.constant dense<0> : tensor<9x3072xi32>
    %360 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_119, %357 : tensor<9x768xi8>, tensor<768x3072xi8>) outs(%cst_123 : tensor<9x3072xi32>) -> tensor<9x3072xi32>
    %361 = tensor.empty() : tensor<9x3072xf32>
    %362 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%360 : tensor<9x3072xi32>) outs(%361 : tensor<9x3072xf32>) {
    ^bb0(%in: i32, %out: f32):
      %481 = arith.sitofp %in : i32 to f32
      linalg.yield %481 : f32
    } -> tensor<9x3072xf32>
    %363 = tensor.empty() : tensor<9x3072xf32>
    %364 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%362, %359 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%363 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x3072xf32>
    %365 = tensor.empty() : tensor<3072xf32>
    %366 = linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel"]} ins(%arg16 : tensor<3072xf16>) outs(%365 : tensor<3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<3072xf32>
    %expanded_124 = tensor.expand_shape %366 [[0, 1]] : tensor<3072xf32> into tensor<1x3072xf32>
    %367 = tensor.empty() : tensor<9x3072xf32>
    %368 = linalg.generic {indexing_maps = [#map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%364, %expanded_124 : tensor<9x3072xf32>, tensor<1x3072xf32>) outs(%367 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x3072xf32>
    %expanded_125 = tensor.expand_shape %368 [[0, 1], [2]] : tensor<9x3072xf32> into tensor<1x9x3072xf32>
    %369 = tensor.empty() : tensor<1x9x3072xf16>
    %370 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_125 : tensor<1x9x3072xf32>) outs(%369 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x3072xf16>
    %expanded_126 = tensor.expand_shape %arg17 [[0, 1, 2]] : tensor<3072xf16> into tensor<1x1x3072xf16>
    %371 = tensor.empty() : tensor<1x9x3072xf16>
    %372 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%370, %expanded_126 : tensor<1x9x3072xf16>, tensor<1x1x3072xf16>) outs(%371 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x3072xf16>
    %373 = tensor.empty() : tensor<1x9x3072xf32>
    %374 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%372 : tensor<1x9x3072xf16>) outs(%373 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %cst_127 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %cst_128 = arith.constant dense<5.000000e-01> : tensor<1x1x1xf32>
    %375 = tensor.empty() : tensor<1x9x3072xf32>
    %376 = linalg.generic {indexing_maps = [#map7, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%374, %cst_128 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%375 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %cst_129 = arith.constant dense<0.707106769> : tensor<1xf32>
    %cst_130 = arith.constant dense<0.707106769> : tensor<1x1x1xf32>
    %377 = tensor.empty() : tensor<1x9x3072xf32>
    %378 = linalg.generic {indexing_maps = [#map7, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%374, %cst_130 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%377 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %379 = math.erf %378 : tensor<1x9x3072xf32>
    %cst_131 = arith.constant dense<1.000000e+00> : tensor<1x9x3072xf32>
    %380 = tensor.empty() : tensor<1x9x3072xf32>
    %381 = linalg.generic {indexing_maps = [#map7, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%379, %cst_131 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%380 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %382 = tensor.empty() : tensor<1x9x3072xf32>
    %383 = linalg.generic {indexing_maps = [#map7, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%376, %381 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%382 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %384 = tensor.empty() : tensor<1x9x3072xf16>
    %385 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%383 : tensor<1x9x3072xf32>) outs(%384 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x3072xf16>
    %386 = tensor.empty() : tensor<1x9xf16>
    %387 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%385 : tensor<1x9x3072xf16>) outs(%386 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.minf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %388 = tensor.empty() : tensor<1x9xf16>
    %389 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%385 : tensor<1x9x3072xf16>) outs(%388 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.maxf %in, %out : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_132 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %390 = tensor.empty() : tensor<1x9xf16>
    %391 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%387, %cst_132 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%390 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.minf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_133 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %392 = tensor.empty() : tensor<1x9xf16>
    %393 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%389, %cst_133 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%392 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %394 = tensor.empty() : tensor<1x9xf16>
    %395 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%391 : tensor<1x9xf16>) outs(%394 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %481 = arith.negf %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %396 = tensor.empty() : tensor<1x9xf16>
    %397 = linalg.generic {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%395, %393 : tensor<1x9xf16>, tensor<1x9xf16>) outs(%396 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.maxf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_134 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %cst_135 = arith.constant dense<1.270000e+02> : tensor<1x1xf16>
    %398 = tensor.empty() : tensor<1x1xf16>
    %399 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_135 : tensor<1x1xf16>) outs(%398 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cst_163 = arith.constant 1.000000e+00 : f16
      %481 = arith.divf %cst_163, %in : f16
      linalg.yield %481 : f16
    } -> tensor<1x1xf16>
    %400 = tensor.empty() : tensor<1x9xf16>
    %401 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%397, %399 : tensor<1x9xf16>, tensor<1x1xf16>) outs(%400 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.mulf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %cst_136 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %402 = tensor.empty() : tensor<1x9xf32>
    %403 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%401 : tensor<1x9xf16>) outs(%402 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %404 = tensor.empty() : tensor<1x9xf32>
    %405 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%403 : tensor<1x9xf32>) outs(%404 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 9.99999974E-6 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9xf32>
    %406 = tensor.empty() : tensor<1x9xf16>
    %407 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%405 : tensor<1x9xf32>) outs(%406 : tensor<1x9xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9xf16>
    %408 = tensor.empty() : tensor<1x9xf32>
    %409 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%407 : tensor<1x9xf16>) outs(%408 : tensor<1x9xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9xf32>
    %expanded_137 = tensor.expand_shape %409 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_138 = arith.constant dense<0.000000e+00> : tensor<1x9x1xf16>
    %410 = tensor.empty() : tensor<1x9x1xf32>
    %411 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_137 : tensor<1x9x1xf32>) outs(%410 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_139 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_140 = arith.constant dense<1.000000e+00> : tensor<1x1x1xf32>
    %412 = tensor.empty() : tensor<1x9x1xf32>
    %413 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%411, %cst_140 : tensor<1x9x1xf32>, tensor<1x1x1xf32>) outs(%412 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %414 = tensor.empty() : tensor<1x9x3072xf32>
    %415 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%385 : tensor<1x9x3072xf16>) outs(%414 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %416 = tensor.empty() : tensor<1x9x3072xf32>
    %417 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%415, %413 : tensor<1x9x3072xf32>, tensor<1x9x1xf32>) outs(%416 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %418 = tensor.empty() : tensor<1x9x3072xf32>
    %419 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%417 : tensor<1x9x3072xf32>) outs(%418 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.floor %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %420 = tensor.empty() : tensor<1x9x1xf32>
    %421 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_138 : tensor<1x9x1xf16>) outs(%420 : tensor<1x9x1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %422 = tensor.empty() : tensor<1x9x3072xf32>
    %423 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%419, %421 : tensor<1x9x3072xf32>, tensor<1x9x1xf32>) outs(%422 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x3072xf32>
    %424 = tensor.empty() : tensor<1x9x3072xf32>
    %425 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%423 : tensor<1x9x3072xf32>) outs(%424 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant -1.270000e+02 : f32
      %cst_164 = arith.constant 0x7F800000 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x3072xf32>
    %426 = tensor.empty() : tensor<1x9x3072xf32>
    %427 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425 : tensor<1x9x3072xf32>) outs(%426 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 0xFF800000 : f32
      %cst_164 = arith.constant 1.270000e+02 : f32
      %481 = arith.minf %in, %cst_164 : f32
      %482 = arith.maxf %481, %cst_163 : f32
      linalg.yield %482 : f32
    } -> tensor<1x9x3072xf32>
    %428 = tensor.empty() : tensor<1x9x3072xi8>
    %429 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%427 : tensor<1x9x3072xf32>) outs(%428 : tensor<1x9x3072xi8>) {
    ^bb0(%in: f32, %out: i8):
      %481 = arith.fptosi %in : f32 to i8
      linalg.yield %481 : i8
    } -> tensor<1x9x3072xi8>
    %cst_141 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %430 = tensor.empty() : tensor<3072x768xi8>
    %431 = linalg.generic {indexing_maps = [#map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg18 : tensor<768x3072xi8>) outs(%430 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<3072x768xi8>
    %collapsed_142 = tensor.collapse_shape %429 [[0, 1], [2]] : tensor<1x9x3072xi8> into tensor<9x3072xi8>
    %collapsed_143 = tensor.collapse_shape %409 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_144 = tensor.expand_shape %collapsed_143 [[0, 1]] : tensor<9xf32> into tensor<9x1xf32>
    %cst_145 = arith.constant dense<0.000000e+00> : tensor<9x768xf32>
    %432 = tensor.empty() : tensor<9x768xf32>
    %433 = linalg.generic {indexing_maps = [#map9, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_144, %cst_145 : tensor<9x1xf32>, tensor<9x768xf32>) outs(%432 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %cst_146 = arith.constant dense<0> : tensor<9x768xi32>
    %434 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_142, %431 : tensor<9x3072xi8>, tensor<3072x768xi8>) outs(%cst_146 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %435 = tensor.empty() : tensor<9x768xf32>
    %436 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%434 : tensor<9x768xi32>) outs(%435 : tensor<9x768xf32>) {
    ^bb0(%in: i32, %out: f32):
      %481 = arith.sitofp %in : i32 to f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %437 = tensor.empty() : tensor<9x768xf32>
    %438 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%436, %433 : tensor<9x768xf32>, tensor<9x768xf32>) outs(%437 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %439 = tensor.empty() : tensor<768xf32>
    %440 = linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel"]} ins(%arg19 : tensor<768xf16>) outs(%439 : tensor<768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<768xf32>
    %expanded_147 = tensor.expand_shape %440 [[0, 1]] : tensor<768xf32> into tensor<1x768xf32>
    %441 = tensor.empty() : tensor<9x768xf32>
    %442 = linalg.generic {indexing_maps = [#map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%438, %expanded_147 : tensor<9x768xf32>, tensor<1x768xf32>) outs(%441 : tensor<9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<9x768xf32>
    %expanded_148 = tensor.expand_shape %442 [[0, 1], [2]] : tensor<9x768xf32> into tensor<1x9x768xf32>
    %443 = tensor.empty() : tensor<1x9x768xf16>
    %444 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_148 : tensor<1x9x768xf32>) outs(%443 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %expanded_149 = tensor.expand_shape %arg20 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %445 = tensor.empty() : tensor<1x9x768xf16>
    %446 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%444, %expanded_149 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%445 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %447 = tensor.empty() : tensor<1x9x768xf16>
    %448 = linalg.generic {indexing_maps = [#map7, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%311, %446 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%447 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_163: f16, %out: f16):
      %481 = arith.addf %in, %in_163 : f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    %449 = tensor.empty() : tensor<1x9x768xf32>
    %450 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%448 : tensor<1x9x768xf16>) outs(%449 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %481 = arith.extf %in : f16 to f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %451 = tensor.empty() : tensor<1x9xf32>
    %cst_150 = arith.constant 0.000000e+00 : f32
    %452 = linalg.fill ins(%cst_150 : f32) outs(%451 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %reduced_151 = linalg.reduce ins(%450 : tensor<1x9x768xf32>) outs(%452 : tensor<1x9xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %481 = arith.addf %in, %init : f32
        linalg.yield %481 : f32
      }
    %expanded_152 = tensor.expand_shape %reduced_151 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_153 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %453 = tensor.empty() : tensor<1xf32>
    %454 = linalg.generic {indexing_maps = [#map16, #map10], iterator_types = ["parallel"]} ins(%cst_153 : tensor<1xf32>) outs(%453 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1xf32>
    %expanded_154 = tensor.expand_shape %454 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %455 = tensor.empty() : tensor<1x9x1xf32>
    %456 = linalg.generic {indexing_maps = [#map6, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_154, %expanded_152 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%455 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %457 = tensor.empty() : tensor<1x9x768xf32>
    %458 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%450, %456 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%457 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.subf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %459 = tensor.empty() : tensor<1x9x768xf32>
    %460 = linalg.generic {indexing_maps = [#map7, #map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%458, %458 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%459 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %461 = tensor.empty() : tensor<1x9xf32>
    %cst_155 = arith.constant 0.000000e+00 : f32
    %462 = linalg.fill ins(%cst_155 : f32) outs(%461 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %reduced_156 = linalg.reduce ins(%460 : tensor<1x9x768xf32>) outs(%462 : tensor<1x9xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %481 = arith.addf %in, %init : f32
        linalg.yield %481 : f32
      }
    %expanded_157 = tensor.expand_shape %reduced_156 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %cst_158 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %463 = tensor.empty() : tensor<1xf32>
    %464 = linalg.generic {indexing_maps = [#map16, #map10], iterator_types = ["parallel"]} ins(%cst_158 : tensor<1xf32>) outs(%463 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_163 = arith.constant 1.000000e+00 : f32
      %481 = arith.divf %cst_163, %in : f32
      linalg.yield %481 : f32
    } -> tensor<1xf32>
    %expanded_159 = tensor.expand_shape %464 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %465 = tensor.empty() : tensor<1x9x1xf32>
    %466 = linalg.generic {indexing_maps = [#map6, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_159, %expanded_157 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%465 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %cst_160 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %467 = tensor.empty() : tensor<1x9x1xf32>
    %468 = linalg.generic {indexing_maps = [#map5, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%466, %cst_160 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%467 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %469 = tensor.empty() : tensor<1x9x1xf32>
    %470 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%468 : tensor<1x9x1xf32>) outs(%469 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %481 = math.rsqrt %in : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x1xf32>
    %471 = tensor.empty() : tensor<1x9x768xf32>
    %472 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%450, %456 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%471 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.subf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %473 = tensor.empty() : tensor<1x9x768xf32>
    %474 = linalg.generic {indexing_maps = [#map7, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%472, %470 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%473 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %expanded_161 = tensor.expand_shape %arg21 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %475 = tensor.empty() : tensor<1x9x768xf32>
    %476 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%474, %expanded_161 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%475 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.mulf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %expanded_162 = tensor.expand_shape %arg22 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %477 = tensor.empty() : tensor<1x9x768xf32>
    %478 = linalg.generic {indexing_maps = [#map7, #map11, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%476, %expanded_162 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%477 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_163: f32, %out: f32):
      %481 = arith.addf %in, %in_163 : f32
      linalg.yield %481 : f32
    } -> tensor<1x9x768xf32>
    %479 = tensor.empty() : tensor<1x9x768xf16>
    %480 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%478 : tensor<1x9x768xf32>) outs(%479 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %481 = arith.truncf %in : f32 to f16
      linalg.yield %481 : f16
    } -> tensor<1x9x768xf16>
    return %480 : tensor<1x9x768xf16>
  }
}

