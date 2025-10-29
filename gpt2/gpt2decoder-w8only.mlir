#map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map5 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<(d0, d1) -> (0, d1)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map13 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {
  func.func @forward(%arg1: tensor<1x9x768xf16>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<2304x768xi8>, %arg5: tensor<2304xf32>, %arg6: tensor<2304xf32>, %arg7: tensor<768x768xi8>, %arg8: tensor<768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<1x9x768xf16>, %arg11: tensor<768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<3072x768xi8>, %arg14: tensor<3072xf32>, %arg15: tensor<3072xf32>, %arg16: tensor<768x3072xi8>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>) -> tensor<1x9x768xf16> {
    %0 = tensor.empty() : tensor<1x9x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10 : tensor<1x9x768xf16>) outs(%0 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %180 = arith.extf %in : f16 to f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %2 = tensor.empty() : tensor<1x9x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced = linalg.reduce ins(%1 : tensor<1x9x768xf32>) outs(%3 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %180 = arith.addf %in, %init : f32
    //     linalg.yield %180 : f32
    //   }
    // %expanded = tensor.expand_shape %reduced [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded = linalg.generic {indexing_maps = [#map1, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : tensor<1x9x768xf32>) outs(%3 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_0 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %4 = tensor.empty() : tensor<1xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_0 : tensor<1xf32>) outs(%4 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_93 = arith.constant 1.000000e+00 : f32
      %180 = arith.divf %cst_93, %in : f32
      linalg.yield %180 : f32
    } -> tensor<1xf32>
    %expanded_1 = tensor.expand_shape %5 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %6 = tensor.empty() : tensor<1x9x1xf32>
    %7 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1, %expanded : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%6 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %8 = tensor.empty() : tensor<1x9x768xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %7 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%8 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.subf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %10 = tensor.empty() : tensor<1x9x768xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %9 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%10 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %12 = tensor.empty() : tensor<1x9x1xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %13 = linalg.fill ins(%cst_2 : f32) outs(%12 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_3 = linalg.reduce ins(%11 : tensor<1x9x768xf32>) outs(%13 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %180 = arith.addf %in, %init : f32
    //     linalg.yield %180 : f32
    //   }
    // %expanded_4 = tensor.expand_shape %reduced_3 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_4 = linalg.generic {indexing_maps = [#map1, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : tensor<1x9x768xf32>) outs(%13 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_5 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %14 = tensor.empty() : tensor<1xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_5 : tensor<1xf32>) outs(%14 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_93 = arith.constant 1.000000e+00 : f32
      %180 = arith.divf %cst_93, %in : f32
      linalg.yield %180 : f32
    } -> tensor<1xf32>
    %expanded_6 = tensor.expand_shape %15 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %16 = tensor.empty() : tensor<1x9x1xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_6, %expanded_4 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%16 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %cst_7 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %18 = tensor.empty() : tensor<1x9x1xf32>
    %19 = linalg.generic {indexing_maps = [#map5, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %cst_7 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%18 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.addf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %20 = tensor.empty() : tensor<1x9x1xf32>
    %21 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19 : tensor<1x9x1xf32>) outs(%20 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %180 = math.rsqrt %in : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %22 = tensor.empty() : tensor<1x9x768xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10 : tensor<1x9x768xf16>) outs(%22 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %180 = arith.extf %in : f16 to f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %24 = tensor.empty() : tensor<1x9x768xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %7 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%24 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.subf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %26 = tensor.empty() : tensor<1x9x768xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %21 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%26 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %expanded_8 = tensor.expand_shape %arg2 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %28 = tensor.empty() : tensor<1x9x768xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %expanded_8 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%28 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %expanded_9 = tensor.expand_shape %arg3 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %30 = tensor.empty() : tensor<1x9x768xf32>
    %31 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %expanded_9 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%30 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.addf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %32 = tensor.empty() : tensor<1x9x768xf16>
    %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<1x9x768xf32>) outs(%32 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<1x9x768xf16>
    %cst_10 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %34 = tensor.empty() : tensor<768x2304xi8>
    %35 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<2304x768xi8>) outs(%34 : tensor<768x2304xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x2304xi8>
    %collapsed = tensor.collapse_shape %33 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %36 = tensor.empty() : tensor<768x2304xf16>
    %37 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%35 : tensor<768x2304xi8>) outs(%36 : tensor<768x2304xf16>) {
    ^bb0(%in: i8, %out: f16):
      %180 = arith.sitofp %in : i8 to f16
      linalg.yield %180 : f16
    } -> tensor<768x2304xf16>
    // %cst_11 = arith.constant dense<0.000000e+00> : tensor<9x2304xf16>
    %181 = tensor.empty() : tensor<9x2304xf16>
    %182 = arith.constant 0.000000e+00 : f16
    %cst_11 = linalg.fill ins(%182 : f16) outs(%181 : tensor<9x2304xf16>) -> tensor<9x2304xf16>
    %38 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed, %37 : tensor<9x768xf16>, tensor<768x2304xf16>) outs(%cst_11 : tensor<9x2304xf16>) -> tensor<9x2304xf16>
    %39 = tensor.empty() : tensor<2304xf16>
    %40 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg5 : tensor<2304xf32>) outs(%39 : tensor<2304xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<2304xf16>
    %expanded_12 = tensor.expand_shape %40 [[0, 1]] : tensor<2304xf16> into tensor<1x2304xf16>
    %41 = tensor.empty() : tensor<9x2304xf16>
    %42 = linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%38, %expanded_12 : tensor<9x2304xf16>, tensor<1x2304xf16>) outs(%41 : tensor<9x2304xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.mulf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<9x2304xf16>
    %expanded_13 = tensor.expand_shape %42 [[0, 1], [2]] : tensor<9x2304xf16> into tensor<1x9x2304xf16>
    %43 = tensor.empty() : tensor<2304xf16>
    %44 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg6 : tensor<2304xf32>) outs(%43 : tensor<2304xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<2304xf16>
    %expanded_14 = tensor.expand_shape %44 [[0, 1, 2]] : tensor<2304xf16> into tensor<1x1x2304xf16>
    %45 = tensor.empty() : tensor<1x9x2304xf16>
    %46 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_13, %expanded_14 : tensor<1x9x2304xf16>, tensor<1x1x2304xf16>) outs(%45 : tensor<1x9x2304xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<1x9x2304xf16>
    %collapsed_15 = tensor.collapse_shape %46 [[0, 1], [2]] : tensor<1x9x2304xf16> into tensor<9x2304xf16>
    %extracted_slice = tensor.extract_slice %46[0, 0, 0] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %extracted_slice_16 = tensor.extract_slice %46[0, 0, 768] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %extracted_slice_17 = tensor.extract_slice %46[0, 0, 1536] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %expanded_18 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_19 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %47 = tensor.empty() : tensor<1x12x9x64xf16>
    %48 = linalg.generic {indexing_maps = [#map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_18 : tensor<1x9x12x64xf16>) outs(%47 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %extracted_slice_20 = tensor.extract_slice %46[0, 0, 0] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %extracted_slice_21 = tensor.extract_slice %46[0, 0, 768] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %extracted_slice_22 = tensor.extract_slice %46[0, 0, 1536] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %expanded_23 = tensor.expand_shape %extracted_slice_21 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_24 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %49 = tensor.empty() : tensor<1x12x9x64xf16>
    %50 = linalg.generic {indexing_maps = [#map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_23 : tensor<1x9x12x64xf16>) outs(%49 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %extracted_slice_25 = tensor.extract_slice %46[0, 0, 0] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %extracted_slice_26 = tensor.extract_slice %46[0, 0, 768] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %extracted_slice_27 = tensor.extract_slice %46[0, 0, 1536] [1, 9, 768] [1, 1, 1] : tensor<1x9x2304xf16> to tensor<1x9x768xf16>
    %expanded_28 = tensor.expand_shape %extracted_slice_27 [[0], [1], [2, 3]] : tensor<1x9x768xf16> into tensor<1x9x12x64xf16>
    %cst_29 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %51 = tensor.empty() : tensor<1x12x9x64xf16>
    %52 = linalg.generic {indexing_maps = [#map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_28 : tensor<1x9x12x64xf16>) outs(%51 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %cst_30 = arith.constant 0.000000e+00 : f16
    %cst_31 = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
    %cst_32 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %53 = tensor.empty() : tensor<1x12x64x9xf16>
    %54 = linalg.generic {indexing_maps = [#map12, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%50 : tensor<1x12x9x64xf16>) outs(%53 : tensor<1x12x64x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x64x9xf16>
    %collapsed_33 = tensor.collapse_shape %48 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %collapsed_34 = tensor.collapse_shape %54 [[0, 1], [2], [3]] : tensor<1x12x64x9xf16> into tensor<12x64x9xf16>
    %cst_35 = arith.constant 0.000000e+00 : f16
    %55 = tensor.empty() : tensor<12x9x9xf16>
    %56 = linalg.fill ins(%cst_35 : f16) outs(%55 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %57 = linalg.batch_matmul ins(%collapsed_33, %collapsed_34 : tensor<12x9x64xf16>, tensor<12x64x9xf16>) outs(%56 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %cst_36 = arith.constant 1.250000e-01 : f16
    %cst_37 = arith.constant dense<1.250000e-01> : tensor<12x9x9xf16>
    %58 = tensor.empty() : tensor<12x9x9xf16>
    %59 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57, %cst_37 : tensor<12x9x9xf16>, tensor<12x9x9xf16>) outs(%58 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.mulf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<12x9x9xf16>
    %expanded_38 = tensor.expand_shape %cst_31 [[0, 1], [2]] : tensor<9x9xf16> into tensor<1x9x9xf16>
    %60 = tensor.empty() : tensor<12x9x9xf16>
    %61 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %expanded_38 : tensor<12x9x9xf16>, tensor<1x9x9xf16>) outs(%60 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<12x9x9xf16>
    %62 = tensor.empty() : tensor<12x9x1xf16>
    %cst_39 = arith.constant -6.550400e+04 : f16
    %63 = linalg.fill ins(%cst_39 : f16) outs(%62 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced_40 = linalg.reduce ins(%61 : tensor<12x9x9xf16>) outs(%63 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %180 = arith.maximumf %in, %init : f16
    //     linalg.yield %180 : f16
    //   }
    // %expanded_41 = tensor.expand_shape %reduced_40 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_41 = linalg.generic {indexing_maps = [#map1, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<12x9x9xf16>) outs(%63 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %64 = tensor.empty() : tensor<12x9x9xf16>
    %65 = linalg.generic {indexing_maps = [#map1, #map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %expanded_41 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%64 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.subf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<12x9x9xf16>
    // %66 = math.exp %65 : tensor<12x9x9xf16>
    %67 = tensor.empty() : tensor<12x9x1xf16>
    %cst_42 = arith.constant 0.000000e+00 : f16
    %68 = linalg.fill ins(%cst_42 : f16) outs(%67 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced_43 = linalg.reduce ins(%66 : tensor<12x9x9xf16>) outs(%68 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %180 = arith.addf %in, %init : f16
    //     linalg.yield %180 : f16
    //   }
    // %expanded_44 = tensor.expand_shape %reduced_43 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_44 = linalg.generic {indexing_maps = [#map1, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<12x9x9xf16>) outs(%68 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %69 = tensor.empty() : tensor<12x9x1xf16>
    %70 = linalg.generic {indexing_maps = [#map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_44 : tensor<12x9x1xf16>) outs(%69 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %180 = math.log %in : f16
      linalg.yield %180 : f16
    } -> tensor<12x9x1xf16>
    %71 = tensor.empty() : tensor<12x9x1xf16>
    %72 = linalg.generic {indexing_maps = [#map13, #map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_41, %70 : tensor<12x9x1xf16>, tensor<12x9x1xf16>) outs(%71 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<12x9x1xf16>
    %73 = tensor.empty() : tensor<12x9x9xf16>
    %74 = linalg.generic {indexing_maps = [#map1, #map13, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %72 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%73 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.subf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<12x9x9xf16>
    // %75 = math.exp %74 : tensor<12x9x9xf16>
    %collapsed_45 = tensor.collapse_shape %72 [[0], [1, 2]] : tensor<12x9x1xf16> into tensor<12x9xf16>
    %expanded_46 = tensor.expand_shape %collapsed_45 [[0, 1], [2]] : tensor<12x9xf16> into tensor<1x12x9xf16>
    %collapsed_47 = tensor.collapse_shape %52 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %cst_48 = arith.constant 0.000000e+00 : f16
    %76 = tensor.empty() : tensor<12x9x64xf16>
    %77 = linalg.fill ins(%cst_48 : f16) outs(%76 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %78 = linalg.batch_matmul ins(%74, %collapsed_47 : tensor<12x9x9xf16>, tensor<12x9x64xf16>) outs(%77 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %expanded_49 = tensor.expand_shape %78 [[0, 1], [2], [3]] : tensor<12x9x64xf16> into tensor<1x12x9x64xf16>
    %cst_50 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %79 = tensor.empty() : tensor<1x9x12x64xf16>
    %80 = linalg.generic {indexing_maps = [#map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_49 : tensor<1x12x9x64xf16>) outs(%79 : tensor<1x9x12x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x9x12x64xf16>
    %collapsed_51 = tensor.collapse_shape %80 [[0], [1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<1x9x768xf16>
    %cst_52 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %81 = tensor.empty() : tensor<768x768xi8>
    %82 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg7 : tensor<768x768xi8>) outs(%81 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x768xi8>
    %collapsed_53 = tensor.collapse_shape %80 [[0, 1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<9x768xf16>
    %83 = tensor.empty() : tensor<768x768xf16>
    %84 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%82 : tensor<768x768xi8>) outs(%83 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %180 = arith.sitofp %in : i8 to f16
      linalg.yield %180 : f16
    } -> tensor<768x768xf16>
    // %cst_54 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %183 = tensor.empty() : tensor<9x768xf16>
    %184 = arith.constant 0.000000e+00 : f16
    %cst_54 = linalg.fill ins(%184 : f16) outs(%183 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %85 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_53, %84 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_54 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %86 = tensor.empty() : tensor<768xf16>
    %87 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg8 : tensor<768xf32>) outs(%86 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<768xf16>
    %expanded_55 = tensor.expand_shape %87 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %88 = tensor.empty() : tensor<9x768xf16>
    %89 = linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%85, %expanded_55 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%88 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.mulf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<9x768xf16>
    %expanded_56 = tensor.expand_shape %89 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %90 = tensor.empty() : tensor<768xf16>
    %91 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg9 : tensor<768xf32>) outs(%90 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<768xf16>
    %expanded_57 = tensor.expand_shape %91 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %92 = tensor.empty() : tensor<1x9x768xf16>
    %93 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_56, %expanded_57 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%92 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_58 = tensor.collapse_shape %93 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %94 = tensor.empty() : tensor<1x9x768xf16>
    %95 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %93 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%94 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<1x9x768xf16>
    %96 = tensor.empty() : tensor<1x9x768xf32>
    %97 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<1x9x768xf16>) outs(%96 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %180 = arith.extf %in : f16 to f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %98 = tensor.empty() : tensor<1x9x1xf32>
    %cst_59 = arith.constant 0.000000e+00 : f32
    %99 = linalg.fill ins(%cst_59 : f32) outs(%98 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_60 = linalg.reduce ins(%97 : tensor<1x9x768xf32>) outs(%99 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %180 = arith.addf %in, %init : f32
    //     linalg.yield %180 : f32
    //   }
    // %expanded_61 = tensor.expand_shape %reduced_60 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_61 = linalg.generic {indexing_maps = [#map1, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%97 : tensor<1x9x768xf32>) outs(%99 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_62 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %100 = tensor.empty() : tensor<1xf32>
    %101 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_62 : tensor<1xf32>) outs(%100 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_93 = arith.constant 1.000000e+00 : f32
      %180 = arith.divf %cst_93, %in : f32
      linalg.yield %180 : f32
    } -> tensor<1xf32>
    %expanded_63 = tensor.expand_shape %101 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %102 = tensor.empty() : tensor<1x9x1xf32>
    %103 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_63, %expanded_61 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%102 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %104 = tensor.empty() : tensor<1x9x768xf32>
    %105 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%97, %103 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%104 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.subf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %106 = tensor.empty() : tensor<1x9x768xf32>
    %107 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105, %105 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%106 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %108 = tensor.empty() : tensor<1x9x1xf32>
    %cst_64 = arith.constant 0.000000e+00 : f32
    %109 = linalg.fill ins(%cst_64 : f32) outs(%108 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_65 = linalg.reduce ins(%107 : tensor<1x9x768xf32>) outs(%109 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %180 = arith.addf %in, %init : f32
    //     linalg.yield %180 : f32
    //   }
    // %expanded_66 = tensor.expand_shape %reduced_65 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_66 = linalg.generic {indexing_maps = [#map1, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%107 : tensor<1x9x768xf32>) outs(%109 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_67 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %110 = tensor.empty() : tensor<1xf32>
    %111 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_67 : tensor<1xf32>) outs(%110 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_93 = arith.constant 1.000000e+00 : f32
      %180 = arith.divf %cst_93, %in : f32
      linalg.yield %180 : f32
    } -> tensor<1xf32>
    %expanded_68 = tensor.expand_shape %111 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %112 = tensor.empty() : tensor<1x9x1xf32>
    %113 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_68, %expanded_66 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%112 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %cst_69 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %114 = tensor.empty() : tensor<1x9x1xf32>
    %115 = linalg.generic {indexing_maps = [#map5, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113, %cst_69 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%114 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.addf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %116 = tensor.empty() : tensor<1x9x1xf32>
    %117 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115 : tensor<1x9x1xf32>) outs(%116 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %180 = math.rsqrt %in : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x1xf32>
    %118 = tensor.empty() : tensor<1x9x768xf32>
    %119 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<1x9x768xf16>) outs(%118 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %180 = arith.extf %in : f16 to f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %120 = tensor.empty() : tensor<1x9x768xf32>
    %121 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %103 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%120 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.subf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %122 = tensor.empty() : tensor<1x9x768xf32>
    %123 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%121, %117 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%122 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %expanded_70 = tensor.expand_shape %arg11 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %124 = tensor.empty() : tensor<1x9x768xf32>
    %125 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %expanded_70 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%124 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %expanded_71 = tensor.expand_shape %arg12 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %126 = tensor.empty() : tensor<1x9x768xf32>
    %127 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %expanded_71 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%126 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.addf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x768xf32>
    %128 = tensor.empty() : tensor<1x9x768xf16>
    %129 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127 : tensor<1x9x768xf32>) outs(%128 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<1x9x768xf16>
    %cst_72 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %130 = tensor.empty() : tensor<768x3072xi8>
    %131 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg13 : tensor<3072x768xi8>) outs(%130 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<768x3072xi8>
    %collapsed_73 = tensor.collapse_shape %129 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %132 = tensor.empty() : tensor<768x3072xf16>
    %133 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%131 : tensor<768x3072xi8>) outs(%132 : tensor<768x3072xf16>) {
    ^bb0(%in: i8, %out: f16):
      %180 = arith.sitofp %in : i8 to f16
      linalg.yield %180 : f16
    } -> tensor<768x3072xf16>
    // %cst_74 = arith.constant dense<0.000000e+00> : tensor<9x3072xf16>
    %185 = tensor.empty() : tensor<9x3072xf16>
    %186 = arith.constant 0.000000e+00 : f16
    %cst_74 = linalg.fill ins(%186 : f16) outs(%185 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %134 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_73, %133 : tensor<9x768xf16>, tensor<768x3072xf16>) outs(%cst_74 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %135 = tensor.empty() : tensor<3072xf16>
    %136 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg14 : tensor<3072xf32>) outs(%135 : tensor<3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<3072xf16>
    %expanded_75 = tensor.expand_shape %136 [[0, 1]] : tensor<3072xf16> into tensor<1x3072xf16>
    %137 = tensor.empty() : tensor<9x3072xf16>
    %138 = linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%134, %expanded_75 : tensor<9x3072xf16>, tensor<1x3072xf16>) outs(%137 : tensor<9x3072xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.mulf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<9x3072xf16>
    %expanded_76 = tensor.expand_shape %138 [[0, 1], [2]] : tensor<9x3072xf16> into tensor<1x9x3072xf16>
    %139 = tensor.empty() : tensor<3072xf16>
    %140 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg15 : tensor<3072xf32>) outs(%139 : tensor<3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<3072xf16>
    %expanded_77 = tensor.expand_shape %140 [[0, 1, 2]] : tensor<3072xf16> into tensor<1x1x3072xf16>
    %141 = tensor.empty() : tensor<1x9x3072xf16>
    %142 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_76, %expanded_77 : tensor<1x9x3072xf16>, tensor<1x1x3072xf16>) outs(%141 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<1x9x3072xf16>
    %collapsed_78 = tensor.collapse_shape %142 [[0, 1], [2]] : tensor<1x9x3072xf16> into tensor<9x3072xf16>
    %143 = tensor.empty() : tensor<1x9x3072xf32>
    %144 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x9x3072xf16>) outs(%143 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %180 = arith.extf %in : f16 to f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %145 = tensor.empty() : tensor<1x9x3072xf32>
    %146 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144, %144 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%145 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %147 = tensor.empty() : tensor<1x9x3072xf32>
    %148 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%146, %144 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%147 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %cst_79 = arith.constant dense<4.471500e-02> : tensor<1xf32>
    %cst_80 = arith.constant dense<4.471500e-02> : tensor<1x1x1xf32>
    %149 = tensor.empty() : tensor<1x9x3072xf32>
    %150 = linalg.generic {indexing_maps = [#map, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%148, %cst_80 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%149 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %151 = tensor.empty() : tensor<1x9x3072xf32>
    %152 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144, %150 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%151 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.addf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %cst_81 = arith.constant dense<0.797884583> : tensor<1xf32>
    %cst_82 = arith.constant dense<0.797884583> : tensor<1x1x1xf32>
    %153 = tensor.empty() : tensor<1x9x3072xf32>
    %154 = linalg.generic {indexing_maps = [#map, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%152, %cst_82 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%153 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %cst_83 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %cst_84 = arith.constant dense<5.000000e-01> : tensor<1x1x1xf32>
    %155 = tensor.empty() : tensor<1x9x3072xf32>
    %156 = linalg.generic {indexing_maps = [#map, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144, %cst_84 : tensor<1x9x3072xf32>, tensor<1x1x1xf32>) outs(%155 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %157 = tensor.empty() : tensor<1x9x3072xf32>
    %158 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%154 : tensor<1x9x3072xf32>) outs(%157 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %180 = math.tanh %in : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %cst_85 = arith.constant dense<1.000000e+00> : tensor<1x9x3072xf32>
    %159 = tensor.empty() : tensor<1x9x3072xf32>
    %160 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%158, %cst_85 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%159 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.addf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %161 = tensor.empty() : tensor<1x9x3072xf32>
    %162 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%156, %160 : tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) outs(%161 : tensor<1x9x3072xf32>) {
    ^bb0(%in: f32, %in_93: f32, %out: f32):
      %180 = arith.mulf %in, %in_93 : f32
      linalg.yield %180 : f32
    } -> tensor<1x9x3072xf32>
    %163 = tensor.empty() : tensor<1x9x3072xf16>
    %164 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x9x3072xf32>) outs(%163 : tensor<1x9x3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<1x9x3072xf16>
    %cst_86 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %165 = tensor.empty() : tensor<3072x768xi8>
    %166 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg16 : tensor<768x3072xi8>) outs(%165 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<3072x768xi8>
    %collapsed_87 = tensor.collapse_shape %164 [[0, 1], [2]] : tensor<1x9x3072xf16> into tensor<9x3072xf16>
    %167 = tensor.empty() : tensor<3072x768xf16>
    %168 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%166 : tensor<3072x768xi8>) outs(%167 : tensor<3072x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %180 = arith.sitofp %in : i8 to f16
      linalg.yield %180 : f16
    } -> tensor<3072x768xf16>
    // %cst_88 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %187 = tensor.empty() : tensor<9x768xf16>
    %188 = arith.constant 0.000000e+00 : f16
    %cst_88 = linalg.fill ins(%188 : f16) outs(%187 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %169 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_87, %168 : tensor<9x3072xf16>, tensor<3072x768xf16>) outs(%cst_88 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %170 = tensor.empty() : tensor<768xf16>
    %171 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg17 : tensor<768xf32>) outs(%170 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<768xf16>
    %expanded_89 = tensor.expand_shape %171 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %172 = tensor.empty() : tensor<9x768xf16>
    %173 = linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%169, %expanded_89 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%172 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.mulf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<9x768xf16>
    %expanded_90 = tensor.expand_shape %173 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %174 = tensor.empty() : tensor<768xf16>
    %175 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg18 : tensor<768xf32>) outs(%174 : tensor<768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %180 = arith.truncf %in : f32 to f16
      linalg.yield %180 : f16
    } -> tensor<768xf16>
    %expanded_91 = tensor.expand_shape %175 [[0, 1, 2]] : tensor<768xf16> into tensor<1x1x768xf16>
    %176 = tensor.empty() : tensor<1x9x768xf16>
    %177 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_90, %expanded_91 : tensor<1x9x768xf16>, tensor<1x1x768xf16>) outs(%176 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_92 = tensor.collapse_shape %177 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %178 = tensor.empty() : tensor<1x9x768xf16>
    %179 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %177 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%178 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_93: f16, %out: f16):
      %180 = arith.addf %in, %in_93 : f16
      linalg.yield %180 : f16
    } -> tensor<1x9x768xf16>
    return %179 : tensor<1x9x768xf16>
  }
}

