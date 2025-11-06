#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map11 = affine_map<(d0) -> (0)>
#map12 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map13 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
module {
  func.func @forward(%arg0: tensor<768x768xi8>, %arg1: tensor<1x9x768xf16>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768x768xi8>, %arg5: tensor<1x9x768xf16>, %arg6: tensor<768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768x768xi8>, %arg9: tensor<1x9x768xf16>, %arg10: tensor<768xf32>, %arg11: tensor<768xf32>, %arg12: tensor<768x768xi8>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<1x9x768xf16>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<3072x768xi8>, %arg19: tensor<3072xf32>, %arg20: tensor<3072xf32>, %arg21: tensor<768x3072xi8>, %arg22: tensor<768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768xf32>) -> tensor<1x9x768xf16> {
    %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
    %0 = tensor.empty() : tensor<768x768xi8>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<768x768xi8>) outs(%0 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %2 = tensor.empty() : tensor<768x768xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<768x768xi8>) outs(%2 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %197 = arith.sitofp %in : i8 to f16
      linalg.yield %197 : f16
    } -> tensor<768x768xf16>
    // %cst_0 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %200 = tensor.empty() : tensor<9x768xf16>
    %cst_201 = arith.constant 0.000000e+00 : f16
    %cst_0 = linalg.fill ins(%cst_201 : f16) outs(%200 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %4 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed, %3 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_0 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %5 = tensor.empty() : tensor<768xf16>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg2 : tensor<768xf32>) outs(%5 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded = tensor.expand_shape %6 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %7 = tensor.empty() : tensor<9x768xf16>
    %8 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %expanded : tensor<9x768xf16>, tensor<1x768xf16>) outs(%7 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<9x768xf16>
    %expanded_1 = tensor.expand_shape %8 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %9 = tensor.empty() : tensor<768xf16>
    %10 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg3 : tensor<768xf32>) outs(%9 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_2 = tensor.expand_shape %10 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %11 = tensor.empty() : tensor<1x9x768xf16>
    %12 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1, %expanded_2 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%11 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_3 = tensor.collapse_shape %12 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %cst_4 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %13 = tensor.empty() : tensor<768x768xi8>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<768x768xi8>) outs(%13 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_5 = tensor.collapse_shape %arg5 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %15 = tensor.empty() : tensor<768x768xf16>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<768x768xi8>) outs(%15 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %197 = arith.sitofp %in : i8 to f16
      linalg.yield %197 : f16
    } -> tensor<768x768xf16>
    // %cst_6 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %202 = tensor.empty() : tensor<9x768xf16>
    %cst_203 = arith.constant 0.000000e+00 : f16
    %cst_6 = linalg.fill ins(%cst_203 : f16) outs(%202 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %17 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_5, %16 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_6 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %18 = tensor.empty() : tensor<768xf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg6 : tensor<768xf32>) outs(%18 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_7 = tensor.expand_shape %19 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %20 = tensor.empty() : tensor<9x768xf16>
    %21 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%17, %expanded_7 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%20 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<9x768xf16>
    %expanded_8 = tensor.expand_shape %21 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %22 = tensor.empty() : tensor<768xf16>
    %23 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg7 : tensor<768xf32>) outs(%22 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_9 = tensor.expand_shape %23 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %24 = tensor.empty() : tensor<1x9x768xf16>
    %25 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_8, %expanded_9 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%24 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_10 = tensor.collapse_shape %25 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %cst_11 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %26 = tensor.empty() : tensor<768x768xi8>
    %27 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<768x768xi8>) outs(%26 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_12 = tensor.collapse_shape %arg9 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %28 = tensor.empty() : tensor<768x768xf16>
    %29 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<768x768xi8>) outs(%28 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %197 = arith.sitofp %in : i8 to f16
      linalg.yield %197 : f16
    } -> tensor<768x768xf16>
    // %cst_13 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %204 = tensor.empty() : tensor<9x768xf16>
    %cst_205 = arith.constant 0.000000e+00 : f16
    %cst_13 = linalg.fill ins(%cst_205 : f16) outs(%204 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %30 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_12, %29 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_13 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %31 = tensor.empty() : tensor<768xf16>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg10 : tensor<768xf32>) outs(%31 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_14 = tensor.expand_shape %32 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %33 = tensor.empty() : tensor<9x768xf16>
    %34 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %expanded_14 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%33 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<9x768xf16>
    %expanded_15 = tensor.expand_shape %34 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %35 = tensor.empty() : tensor<768xf16>
    %36 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg11 : tensor<768xf32>) outs(%35 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_16 = tensor.expand_shape %36 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %37 = tensor.empty() : tensor<1x9x768xf16>
    %38 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_15, %expanded_16 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%37 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_17 = tensor.collapse_shape %38 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %expanded_18 = tensor.expand_shape %12 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_19 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %39 = tensor.empty() : tensor<1x12x9x64xf16>
    %40 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_18 : tensor<1x9x12x64xf16>) outs(%39 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %expanded_20 = tensor.expand_shape %25 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_21 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %41 = tensor.empty() : tensor<1x12x9x64xf16>
    %42 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_20 : tensor<1x9x12x64xf16>) outs(%41 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %expanded_22 = tensor.expand_shape %38 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_23 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %43 = tensor.empty() : tensor<1x12x9x64xf16>
    %44 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_22 : tensor<1x9x12x64xf16>) outs(%43 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %cst_24 = arith.constant 0.000000e+00 : f16
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
    %cst_26 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %45 = tensor.empty() : tensor<1x12x64x9xf16>
    %46 = linalg.generic {indexing_maps = [#map9, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<1x12x9x64xf16>) outs(%45 : tensor<1x12x64x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x64x9xf16>
    %collapsed_27 = tensor.collapse_shape %40 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %collapsed_28 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<1x12x64x9xf16> into tensor<12x64x9xf16>
    %cst_29 = arith.constant 0.000000e+00 : f16
    %47 = tensor.empty() : tensor<12x9x9xf16>
    %48 = linalg.fill ins(%cst_29 : f16) outs(%47 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %49 = linalg.batch_matmul ins(%collapsed_27, %collapsed_28 : tensor<12x9x64xf16>, tensor<12x64x9xf16>) outs(%48 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %cst_30 = arith.constant 1.250000e-01 : f16
    %cst_31 = arith.constant dense<1.250000e-01> : tensor<12x9x9xf16>
    %50 = tensor.empty() : tensor<12x9x9xf16>
    %51 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %cst_31 : tensor<12x9x9xf16>, tensor<12x9x9xf16>) outs(%50 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x9xf16>
    %expanded_32 = tensor.expand_shape %cst_25 [[0, 1], [2]] : tensor<9x9xf16> into tensor<1x9x9xf16>
    %52 = tensor.empty() : tensor<12x9x9xf16>
    %53 = linalg.generic {indexing_maps = [#map6, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %expanded_32 : tensor<12x9x9xf16>, tensor<1x9x9xf16>) outs(%52 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x9xf16>
    %54 = tensor.empty() : tensor<12x9x1xf16>
    %cst_33 = arith.constant -6.550400e+04 : f16
    %55 = linalg.fill ins(%cst_33 : f16) outs(%54 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced = linalg.reduce ins(%53 : tensor<12x9x9xf16>) outs(%55 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %197 = arith.addf %in, %init : f16
    //     linalg.yield %197 : f16
    //   }
    // %expanded_34 = tensor.expand_shape %reduced [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_34 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%53 : tensor<12x9x9xf16>) outs(%55 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %56 = tensor.empty() : tensor<12x9x9xf16>
    %57 = linalg.generic {indexing_maps = [#map6, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53, %expanded_34 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%56 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.subf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x9xf16>
    // %58 = math.exp %57 : tensor<12x9x9xf16>
    %_58 = tensor.empty() : tensor<12x9x9xf16>
    %58 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<12x9x9xf16>) outs(%_58 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %197 = math.exp %in : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x9xf16>
    %59 = tensor.empty() : tensor<12x9x1xf16>
    %cst_35 = arith.constant 0.000000e+00 : f16
    %60 = linalg.fill ins(%cst_35 : f16) outs(%59 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced_36 = linalg.reduce ins(%58 : tensor<12x9x9xf16>) outs(%60 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %197 = arith.addf %in, %init : f16
    //     linalg.yield %197 : f16
    //   }
    // %expanded_37 = tensor.expand_shape %reduced_36 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_37 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%58 : tensor<12x9x9xf16>) outs(%60 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %61 = tensor.empty() : tensor<12x9x1xf16>
    %62 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37 : tensor<12x9x1xf16>) outs(%61 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %197 = math.log %in : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x1xf16>
    %63 = tensor.empty() : tensor<12x9x1xf16>
    %64 = linalg.generic {indexing_maps = [#map10, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_34, %62 : tensor<12x9x1xf16>, tensor<12x9x1xf16>) outs(%63 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x1xf16>
    %65 = tensor.empty() : tensor<12x9x9xf16>
    %66 = linalg.generic {indexing_maps = [#map6, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53, %64 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%65 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.subf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x9xf16>
    // %67 = math.exp %66 : tensor<12x9x9xf16>
    %_67 = tensor.empty() : tensor<12x9x9xf16>
    %67 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%66 : tensor<12x9x9xf16>) outs(%_67 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %197 = math.exp %in : f16
      linalg.yield %197 : f16
    } -> tensor<12x9x9xf16>
    %collapsed_38 = tensor.collapse_shape %64 [[0], [1, 2]] : tensor<12x9x1xf16> into tensor<12x9xf16>
    %expanded_39 = tensor.expand_shape %collapsed_38 [[0, 1], [2]] : tensor<12x9xf16> into tensor<1x12x9xf16>
    %collapsed_40 = tensor.collapse_shape %44 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %cst_41 = arith.constant 0.000000e+00 : f16
    %68 = tensor.empty() : tensor<12x9x64xf16>
    %69 = linalg.fill ins(%cst_41 : f16) outs(%68 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %70 = linalg.batch_matmul ins(%67, %collapsed_40 : tensor<12x9x9xf16>, tensor<12x9x64xf16>) outs(%69 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %expanded_42 = tensor.expand_shape %70 [[0, 1], [2], [3]] : tensor<12x9x64xf16> into tensor<1x12x9x64xf16>
    %cst_43 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %71 = tensor.empty() : tensor<1x9x12x64xf16>
    %72 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_42 : tensor<1x12x9x64xf16>) outs(%71 : tensor<1x9x12x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x9x12x64xf16>
    %collapsed_44 = tensor.collapse_shape %72 [[0], [1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<1x9x768xf16>
    %cst_45 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %73 = tensor.empty() : tensor<768x768xi8>
    %74 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg12 : tensor<768x768xi8>) outs(%73 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_46 = tensor.collapse_shape %72 [[0, 1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<9x768xf16>
    %75 = tensor.empty() : tensor<768x768xf16>
    %76 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%74 : tensor<768x768xi8>) outs(%75 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %197 = arith.sitofp %in : i8 to f16
      linalg.yield %197 : f16
    } -> tensor<768x768xf16>
    // %cst_47 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %207 = tensor.empty() : tensor<9x768xf16>
    %cst_208 = arith.constant 0.000000e+00 : f16
    %cst_47 = linalg.fill ins(%cst_208 : f16) outs(%207 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %77 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_46, %76 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_47 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %78 = tensor.empty() : tensor<768xf16>
    %79 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg13 : tensor<768xf32>) outs(%78 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_48 = tensor.expand_shape %79 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %80 = tensor.empty() : tensor<9x768xf16>
    %81 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%77, %expanded_48 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%80 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<9x768xf16>
    %expanded_49 = tensor.expand_shape %81 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %82 = tensor.empty() : tensor<768xf16>
    %83 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg14 : tensor<768xf32>) outs(%82 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_50 = tensor.expand_shape %83 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %84 = tensor.empty() : tensor<1x9x768xf16>
    %85 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_49, %expanded_50 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%84 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_51 = tensor.collapse_shape %85 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %86 = tensor.empty() : tensor<1x9x768xf16>
    %87 = linalg.generic {indexing_maps = [#map4, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %arg15 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%86 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %88 = tensor.empty() : tensor<1x9x768xf32>
    %89 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<1x9x768xf16>) outs(%88 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %197 = arith.extf %in : f16 to f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %90 = tensor.empty() : tensor<1x9x1xf32>
    %cst_52 = arith.constant 0.000000e+00 : f32
    %91 = linalg.fill ins(%cst_52 : f32) outs(%90 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_53 = linalg.reduce ins(%89 : tensor<1x9x768xf32>) outs(%91 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %197 = arith.addf %in, %init : f32
    //     linalg.yield %197 : f32
    //   }
    // %expanded_54 = tensor.expand_shape %reduced_53 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_54 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%89 : tensor<1x9x768xf32>) outs(%91 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_55 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %92 = tensor.empty() : tensor<1xf32>
    %93 = linalg.generic {indexing_maps = [#map11, #map2], iterator_types = ["parallel"]} ins(%cst_55 : tensor<1xf32>) outs(%92 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_97 = arith.constant 1.000000e+00 : f32
      %197 = arith.divf %cst_97, %in : f32
      linalg.yield %197 : f32
    } -> tensor<1xf32>
    %expanded_56 = tensor.expand_shape %93 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %94 = tensor.empty() : tensor<1x9x1xf32>
    %95 = linalg.generic {indexing_maps = [#map12, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_56, %expanded_54 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%94 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %96 = tensor.empty() : tensor<1x9x768xf32>
    %97 = linalg.generic {indexing_maps = [#map4, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%96 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.subf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %98 = tensor.empty() : tensor<1x9x768xf32>
    %99 = linalg.generic {indexing_maps = [#map4, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%97, %97 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%98 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %100 = tensor.empty() : tensor<1x9x1xf32>
    %cst_57 = arith.constant 0.000000e+00 : f32
    %101 = linalg.fill ins(%cst_57 : f32) outs(%100 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_58 = linalg.reduce ins(%99 : tensor<1x9x768xf32>) outs(%101 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %197 = arith.addf %in, %init : f32
    //     linalg.yield %197 : f32
    //   }
    // %expanded_59 = tensor.expand_shape %reduced_58 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_59 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<1x9x768xf32>) outs(%101 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_60 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %102 = tensor.empty() : tensor<1xf32>
    %103 = linalg.generic {indexing_maps = [#map11, #map2], iterator_types = ["parallel"]} ins(%cst_60 : tensor<1xf32>) outs(%102 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_97 = arith.constant 1.000000e+00 : f32
      %197 = arith.divf %cst_97, %in : f32
      linalg.yield %197 : f32
    } -> tensor<1xf32>
    %expanded_61 = tensor.expand_shape %103 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %104 = tensor.empty() : tensor<1x9x1xf32>
    %105 = linalg.generic {indexing_maps = [#map12, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_61, %expanded_59 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%104 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %cst_62 = arith.constant dense<9.99999996E-13> : tensor<1x9x1xf32>
    %106 = tensor.empty() : tensor<1x9x1xf32>
    %107 = linalg.generic {indexing_maps = [#map13, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105, %cst_62 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%106 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.addf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %108 = tensor.empty() : tensor<1x9x1xf32>
    %109 = linalg.generic {indexing_maps = [#map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%107 : tensor<1x9x1xf32>) outs(%108 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %197 = math.rsqrt %in : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %110 = tensor.empty() : tensor<1x9x768xf32>
    %111 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<1x9x768xf16>) outs(%110 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %197 = arith.extf %in : f16 to f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %112 = tensor.empty() : tensor<1x9x768xf32>
    %113 = linalg.generic {indexing_maps = [#map4, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%111, %95 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%112 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.subf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %114 = tensor.empty() : tensor<1x9x768xf32>
    %115 = linalg.generic {indexing_maps = [#map4, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113, %109 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%114 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %expanded_63 = tensor.expand_shape %arg16 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %116 = tensor.empty() : tensor<1x9x768xf32>
    %117 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %expanded_63 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%116 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %expanded_64 = tensor.expand_shape %arg17 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %118 = tensor.empty() : tensor<1x9x768xf32>
    %119 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %expanded_64 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%118 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.addf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %120 = tensor.empty() : tensor<1x9x768xf16>
    %121 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119 : tensor<1x9x768xf32>) outs(%120 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %cst_65 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %122 = tensor.empty() : tensor<768x3072xi8>
    %123 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg18 : tensor<3072x768xi8>) outs(%122 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x3072xi8>
    %collapsed_66 = tensor.collapse_shape %121 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %124 = tensor.empty() : tensor<768x3072xf16>
    %125 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%123 : tensor<768x3072xi8>) outs(%124 : tensor<768x3072xf16>) {
    ^bb0(%in: i8, %out: f16):
      %197 = arith.sitofp %in : i8 to f16
      linalg.yield %197 : f16
    } -> tensor<768x3072xf16>
    // %cst_67 = arith.constant dense<0.000000e+00> : tensor<9x3072xf16>
    %209 = tensor.empty() : tensor<9x3072xf16>
    %cst_210 = arith.constant 0.000000e+00 : f16
    %cst_67 = linalg.fill ins(%cst_210 : f16) outs(%209 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %126 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_66, %125 : tensor<9x768xf16>, tensor<768x3072xf16>) outs(%cst_67 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %127 = tensor.empty() : tensor<3072xf16>
    %128 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg19 : tensor<3072xf32>) outs(%127 : tensor<3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<3072xf16>
    %expanded_68 = tensor.expand_shape %128 [[0, 1]] : tensor<3072xf16> into tensor<1x3072xf16>
    %129 = tensor.empty() : tensor<9x3072xf16>
    %130 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%126, %expanded_68 : tensor<9x3072xf16>, tensor<1x3072xf16>) outs(%129 : tensor<9x3072xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<9x3072xf16>
    %expanded_69 = tensor.expand_shape %130 [[0, 1], [2]] : tensor<9x3072xf16> into tensor<1x9x3072xf16>
    %131 = tensor.empty() : tensor<3072xf16>
    %132 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg20 : tensor<3072xf32>) outs(%131 : tensor<3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<3072xf16>
    %expanded_70 = tensor.expand_shape %132 [[0, 1, 2]] : tensor<3072xf16> into tensor<1x1x3072xf16>
    %133 = tensor.empty() : tensor<1x9x3072xf16>
    %134 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_69, %expanded_70 : tensor<1x9x3072xf16>, tensor<1x1x3072xf16>) outs(%133 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x3072xf16>
    %collapsed_71 = tensor.collapse_shape %134 [[0, 1], [2]] : tensor<1x9x3072xf16> into tensor<9x3072xf16>
    %135 = tensor.empty() : tensor<1x9x3072xf32>
    %136 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%134 : tensor<1x9x3072xf16>) outs(%135 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %197 = arith.extf %in : f16 to f32
      linalg.yield %197 : f32
    } -> tensor<1x9x3072xf32>
    %cst_72 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %cst_73 = arith.constant dense<5.000000e-01> : tensor<1x1x1xf32>
    %137 = tensor.empty() : tensor<1x9x3072xf32>
    %138 = linalg.generic {indexing_maps = [#map4, #map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%136, %cst_73 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%137 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x3072xf32>
    %cst_74 = arith.constant dense<0.707106769> : tensor<1xf32>
    %cst_75 = arith.constant dense<0.707106769> : tensor<1x1x1xf32>
    %139 = tensor.empty() : tensor<1x9x3072xf32>
    %140 = linalg.generic {indexing_maps = [#map4, #map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%136, %cst_75 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%139 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x3072xf32>
    // %141 = math.erf %140 : tensor<1x9x3072xf32>
    %cst_76 = arith.constant dense<1.000000e+00> : tensor<1x9x3072xf32>
    %142 = tensor.empty() : tensor<1x9x3072xf32>
    %143 = linalg.generic {indexing_maps = [#map4, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %cst_76 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%142 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.addf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x3072xf32>
    %144 = tensor.empty() : tensor<1x9x3072xf32>
    %145 = linalg.generic {indexing_maps = [#map4, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%138, %143 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%144 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x3072xf32>
    %146 = tensor.empty() : tensor<1x9x3072xf16>
    %147 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%145 : tensor<1x9x3072xf32>) outs(%146 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<1x9x3072xf16>
    %cst_77 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %148 = tensor.empty() : tensor<3072x768xi8>
    %149 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg21 : tensor<768x3072xi8>) outs(%148 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<3072x768xi8>
    %collapsed_78 = tensor.collapse_shape %147 [[0, 1], [2]] : tensor<1x9x3072xf16> into tensor<9x3072xf16>
    %150 = tensor.empty() : tensor<3072x768xf16>
    %151 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%149 : tensor<3072x768xi8>) outs(%150 : tensor<3072x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %197 = arith.sitofp %in : i8 to f16
      linalg.yield %197 : f16
    } -> tensor<3072x768xf16>
    // %cst_79 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %211 = tensor.empty() : tensor<9x768xf16>
    %cst_212 = arith.constant 0.000000e+00 : f16
    %cst_79 = linalg.fill ins(%cst_212 : f16) outs(%211 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %152 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_78, %151 : tensor<9x3072xf16>, tensor<3072x768xf16>) outs(%cst_79 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %153 = tensor.empty() : tensor<768xf16>
    %154 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg22 : tensor<768xf32>) outs(%153 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_80 = tensor.expand_shape %154 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %155 = tensor.empty() : tensor<9x768xf16>
    %156 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%152, %expanded_80 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%155 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.mulf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<9x768xf16>
    %expanded_81 = tensor.expand_shape %156 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %157 = tensor.empty() : tensor<768xf16>
    %158 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg23 : tensor<768xf32>) outs(%157 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<768xf16>
    %expanded_82 = tensor.expand_shape %158 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %159 = tensor.empty() : tensor<1x9x768xf16>
    %160 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_81, %expanded_82 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%159 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_83 = tensor.collapse_shape %160 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %161 = tensor.empty() : tensor<1x9x768xf16>
    %162 = linalg.generic {indexing_maps = [#map4, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160, %121 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%161 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_97: f16, %out: f16):
      %197 = arith.addf %in, %in_97 : f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    %163 = tensor.empty() : tensor<1x9x768xf32>
    %164 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x9x768xf16>) outs(%163 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %197 = arith.extf %in : f16 to f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %165 = tensor.empty() : tensor<1x9x1xf32>
    %cst_84 = arith.constant 0.000000e+00 : f32
    %166 = linalg.fill ins(%cst_84 : f32) outs(%165 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_85 = linalg.reduce ins(%164 : tensor<1x9x768xf32>) outs(%166 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %197 = arith.addf %in, %init : f32
    //     linalg.yield %197 : f32
    //   }
    // %expanded_86 = tensor.expand_shape %reduced_85 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_86 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%164 : tensor<1x9x768xf32>) outs(%166 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_87 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %167 = tensor.empty() : tensor<1xf32>
    %168 = linalg.generic {indexing_maps = [#map11, #map2], iterator_types = ["parallel"]} ins(%cst_87 : tensor<1xf32>) outs(%167 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_97 = arith.constant 1.000000e+00 : f32
      %197 = arith.divf %cst_97, %in : f32
      linalg.yield %197 : f32
    } -> tensor<1xf32>
    %expanded_88 = tensor.expand_shape %168 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %169 = tensor.empty() : tensor<1x9x1xf32>
    %170 = linalg.generic {indexing_maps = [#map12, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_88, %expanded_86 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%169 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %171 = tensor.empty() : tensor<1x9x768xf32>
    %172 = linalg.generic {indexing_maps = [#map4, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%164, %170 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%171 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.subf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %173 = tensor.empty() : tensor<1x9x768xf32>
    %174 = linalg.generic {indexing_maps = [#map4, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172, %172 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%173 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %175 = tensor.empty() : tensor<1x9x1xf32>
    %cst_89 = arith.constant 0.000000e+00 : f32
    %176 = linalg.fill ins(%cst_89 : f32) outs(%175 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_90 = linalg.reduce ins(%174 : tensor<1x9x768xf32>) outs(%176 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %197 = arith.addf %in, %init : f32
    //     linalg.yield %197 : f32
    //   }
    // %expanded_91 = tensor.expand_shape %reduced_90 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_91 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%174 : tensor<1x9x768xf32>) outs(%176 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_92 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %177 = tensor.empty() : tensor<1xf32>
    %178 = linalg.generic {indexing_maps = [#map11, #map2], iterator_types = ["parallel"]} ins(%cst_92 : tensor<1xf32>) outs(%177 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_97 = arith.constant 1.000000e+00 : f32
      %197 = arith.divf %cst_97, %in : f32
      linalg.yield %197 : f32
    } -> tensor<1xf32>
    %expanded_93 = tensor.expand_shape %178 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %179 = tensor.empty() : tensor<1x9x1xf32>
    %180 = linalg.generic {indexing_maps = [#map12, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_93, %expanded_91 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%179 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %cst_94 = arith.constant dense<9.99999996E-13> : tensor<1x9x1xf32>
    %181 = tensor.empty() : tensor<1x9x1xf32>
    %182 = linalg.generic {indexing_maps = [#map13, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180, %cst_94 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%181 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.addf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %183 = tensor.empty() : tensor<1x9x1xf32>
    %184 = linalg.generic {indexing_maps = [#map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%182 : tensor<1x9x1xf32>) outs(%183 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %197 = math.rsqrt %in : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x1xf32>
    %185 = tensor.empty() : tensor<1x9x768xf32>
    %186 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x9x768xf16>) outs(%185 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %197 = arith.extf %in : f16 to f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %187 = tensor.empty() : tensor<1x9x768xf32>
    %188 = linalg.generic {indexing_maps = [#map4, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%186, %170 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%187 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.subf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %189 = tensor.empty() : tensor<1x9x768xf32>
    %190 = linalg.generic {indexing_maps = [#map4, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %184 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%189 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %expanded_95 = tensor.expand_shape %arg24 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %191 = tensor.empty() : tensor<1x9x768xf32>
    %192 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190, %expanded_95 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%191 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.mulf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %expanded_96 = tensor.expand_shape %arg25 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %193 = tensor.empty() : tensor<1x9x768xf32>
    %194 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192, %expanded_96 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%193 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_97: f32, %out: f32):
      %197 = arith.addf %in, %in_97 : f32
      linalg.yield %197 : f32
    } -> tensor<1x9x768xf32>
    %195 = tensor.empty() : tensor<1x9x768xf16>
    %196 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%194 : tensor<1x9x768xf32>) outs(%195 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %197 = arith.truncf %in : f32 to f16
      linalg.yield %197 : f16
    } -> tensor<1x9x768xf16>
    return %196 : tensor<1x9x768xf16>
  }
}
