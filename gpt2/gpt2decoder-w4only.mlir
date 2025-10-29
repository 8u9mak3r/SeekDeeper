#map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map5 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1) -> (d0, d1 * 2)>
#map9 = affine_map<(d0, d1) -> (d0, d1 * 2 + 1)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map12 = affine_map<(d0, d1) -> (d1, d0)>
#map13 = affine_map<(d0, d1) -> (0, d1)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map17 = affine_map<(d0, d1) -> (0, 0)>
module {
  func.func @forward(%arg0: tensor<1x9x768xf16>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<2304x384xi8>, %arg4: tensor<6x2304x2xf16>, %arg5: tensor<2304xf16>, %arg6: tensor<768x384xi8>, %arg7: tensor<6x768x2xf16>, %arg8: tensor<768xf16>, %arg9: tensor<1x9x768xf16>, %arg10: tensor<768xf32>, %arg11: tensor<768xf32>, %arg12: tensor<3072x384xi8>, %arg13: tensor<6x3072x2xf16>, %arg14: tensor<3072xf16>, %arg15: tensor<768x1536xi8>, %arg16: tensor<24x768x2xf16>, %arg17: tensor<768xf16>) -> tensor<1x9x768xf16> {
    %0 = tensor.empty() : tensor<1x9x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9 : tensor<1x9x768xf16>) outs(%0 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %188 = arith.extf %in : f16 to f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %2 = tensor.empty() : tensor<1x9x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced = linalg.reduce ins(%1 : tensor<1x9x768xf32>) outs(%3 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %188 = arith.addf %in, %init : f32
    //     linalg.yield %188 : f32
    //   }
    // %expanded = tensor.expand_shape %reduced [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : tensor<1x9x768xf32>) outs(%3 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_0 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %4 = tensor.empty() : tensor<1xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_0 : tensor<1xf32>) outs(%4 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_96 = arith.constant 1.000000e+00 : f32
      %188 = arith.divf %cst_96, %in : f32
      linalg.yield %188 : f32
    } -> tensor<1xf32>
    %expanded_1 = tensor.expand_shape %5 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %6 = tensor.empty() : tensor<1x9x1xf32>
    %7 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1, %expanded : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%6 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %8 = tensor.empty() : tensor<1x9x768xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %7 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%8 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.subf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %10 = tensor.empty() : tensor<1x9x768xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %9 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%10 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %12 = tensor.empty() : tensor<1x9x1xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %13 = linalg.fill ins(%cst_2 : f32) outs(%12 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_3 = linalg.reduce ins(%11 : tensor<1x9x768xf32>) outs(%13 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %188 = arith.addf %in, %init : f32
    //     linalg.yield %188 : f32
    //   }
    // %expanded_4 = tensor.expand_shape %reduced_3 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_4 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : tensor<1x9x768xf32>) outs(%13 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_5 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %14 = tensor.empty() : tensor<1xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_5 : tensor<1xf32>) outs(%14 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_96 = arith.constant 1.000000e+00 : f32
      %188 = arith.divf %cst_96, %in : f32
      linalg.yield %188 : f32
    } -> tensor<1xf32>
    %expanded_6 = tensor.expand_shape %15 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %16 = tensor.empty() : tensor<1x9x1xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_6, %expanded_4 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%16 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %cst_7 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %18 = tensor.empty() : tensor<1x9x1xf32>
    %19 = linalg.generic {indexing_maps = [#map5, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %cst_7 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%18 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.addf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %20 = tensor.empty() : tensor<1x9x1xf32>
    %21 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19 : tensor<1x9x1xf32>) outs(%20 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %188 = math.rsqrt %in : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %22 = tensor.empty() : tensor<1x9x768xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %7 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%22 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.subf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %24 = tensor.empty() : tensor<1x9x768xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %21 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%24 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %expanded_8 = tensor.expand_shape %arg1 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %26 = tensor.empty() : tensor<1x9x768xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %expanded_8 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%26 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %expanded_9 = tensor.expand_shape %arg2 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %28 = tensor.empty() : tensor<1x9x768xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %expanded_9 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%28 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.addf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %30 = tensor.empty() : tensor<1x9x768xf16>
    %31 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29 : tensor<1x9x768xf32>) outs(%30 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %188 = arith.truncf %in : f32 to f16
      linalg.yield %188 : f16
    } -> tensor<1x9x768xf16>
    %collapsed = tensor.collapse_shape %31 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %32 = tensor.empty() : tensor<2304x768xi8>
    %33 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<2304x384xi8>) outs(%32 : tensor<2304x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.shrui %in, %c4_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<2304x768xi8>
    %34 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<2304x384xi8>) outs(%33 : tensor<2304x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.andi %in, %c15_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<2304x768xi8>
    %cst_10 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %35 = tensor.empty() : tensor<2304x6x2xf16>
    %36 = linalg.generic {indexing_maps = [#map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4 : tensor<6x2304x2xf16>) outs(%35 : tensor<2304x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<2304x6x2xf16>
    %extracted_slice = tensor.extract_slice %36[0, 0, 0] [2304, 6, 1] [1, 1, 1] : tensor<2304x6x2xf16> to tensor<2304x6x1xf16>
    %extracted_slice_11 = tensor.extract_slice %36[0, 0, 1] [2304, 6, 1] [1, 1, 1] : tensor<2304x6x2xf16> to tensor<2304x6x1xf16>
    %37 = tensor.empty() : tensor<2304x768xf16>
    %38 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<2304x768xi8>) outs(%37 : tensor<2304x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %188 = arith.sitofp %in : i8 to f16
      linalg.yield %188 : f16
    } -> tensor<2304x768xf16>
    %expanded_12 = tensor.expand_shape %38 [[0], [1, 2]] : tensor<2304x768xf16> into tensor<2304x6x128xf16>
    %39 = tensor.empty() : tensor<2304x6x128xf16>
    %40 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_12, %extracted_slice : tensor<2304x6x128xf16>, tensor<2304x6x1xf16>) outs(%39 : tensor<2304x6x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.mulf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<2304x6x128xf16>
    %41 = tensor.empty() : tensor<2304x6x128xf16>
    %42 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %extracted_slice_11 : tensor<2304x6x128xf16>, tensor<2304x6x1xf16>) outs(%41 : tensor<2304x6x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<2304x6x128xf16>
    %collapsed_13 = tensor.collapse_shape %42 [[0], [1, 2]] : tensor<2304x6x128xf16> into tensor<2304x768xf16>
    %cst_14 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %43 = tensor.empty() : tensor<768x2304xf16>
    %44 = linalg.generic {indexing_maps = [#map12, #map7], iterator_types = ["parallel", "parallel"]} ins(%collapsed_13 : tensor<2304x768xf16>) outs(%43 : tensor<768x2304xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x2304xf16>
    // %cst_15 = arith.constant dense<0.000000e+00> : tensor<9x2304xf16>
    %200 = tensor.empty() : tensor<9x2304xf16>
    %201 = arith.constant 0.000000e+00 : f16
    %cst_15 = linalg.fill ins(%201 : f16) outs(%200 : tensor<9x2304xf16>) -> tensor<9x2304xf16>
    %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed, %44 : tensor<9x768xf16>, tensor<768x2304xf16>) outs(%cst_15 : tensor<9x2304xf16>) -> tensor<9x2304xf16>
    %expanded_16 = tensor.expand_shape %arg5 [[0, 1]] : tensor<2304xf16> into tensor<1x2304xf16>
    %46 = tensor.empty() : tensor<9x2304xf16>
    %47 = linalg.generic {indexing_maps = [#map7, #map13, #map7], iterator_types = ["parallel", "parallel"]} ins(%45, %expanded_16 : tensor<9x2304xf16>, tensor<1x2304xf16>) outs(%46 : tensor<9x2304xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<9x2304xf16>
    %extracted_slice_17 = tensor.extract_slice %47[0, 0] [9, 768] [1, 1] : tensor<9x2304xf16> to tensor<9x768xf16>
    %expanded_18 = tensor.expand_shape %extracted_slice_17 [[0, 1], [2, 3]] : tensor<9x768xf16> into tensor<1x9x12x64xf16>
    %cst_19 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %48 = tensor.empty() : tensor<1x12x9x64xf16>
    %49 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_18 : tensor<1x9x12x64xf16>) outs(%48 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %extracted_slice_20 = tensor.extract_slice %47[0, 768] [9, 768] [1, 1] : tensor<9x2304xf16> to tensor<9x768xf16>
    %expanded_21 = tensor.expand_shape %extracted_slice_20 [[0, 1], [2, 3]] : tensor<9x768xf16> into tensor<1x9x12x64xf16>
    %cst_22 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %50 = tensor.empty() : tensor<1x12x9x64xf16>
    %51 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21 : tensor<1x9x12x64xf16>) outs(%50 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %extracted_slice_23 = tensor.extract_slice %47[0, 1536] [9, 768] [1, 1] : tensor<9x2304xf16> to tensor<9x768xf16>
    %expanded_24 = tensor.expand_shape %extracted_slice_23 [[0, 1], [2, 3]] : tensor<9x768xf16> into tensor<1x9x12x64xf16>
    %cst_25 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %52 = tensor.empty() : tensor<1x12x9x64xf16>
    %53 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<1x9x12x64xf16>) outs(%52 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %cst_26 = arith.constant 0.000000e+00 : f16
    %cst_27 = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
    %cst_28 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %54 = tensor.empty() : tensor<1x12x64x9xf16>
    %55 = linalg.generic {indexing_maps = [#map16, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51 : tensor<1x12x9x64xf16>) outs(%54 : tensor<1x12x64x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x64x9xf16>
    %collapsed_29 = tensor.collapse_shape %49 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %collapsed_30 = tensor.collapse_shape %55 [[0, 1], [2], [3]] : tensor<1x12x64x9xf16> into tensor<12x64x9xf16>
    %cst_31 = arith.constant 0.000000e+00 : f16
    %56 = tensor.empty() : tensor<12x9x9xf16>
    %57 = linalg.fill ins(%cst_31 : f16) outs(%56 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %58 = linalg.batch_matmul ins(%collapsed_29, %collapsed_30 : tensor<12x9x64xf16>, tensor<12x64x9xf16>) outs(%57 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %cst_32 = arith.constant 1.250000e-01 : f16
    %cst_33 = arith.constant dense<1.250000e-01> : tensor<12x9x9xf16>
    %59 = tensor.empty() : tensor<12x9x9xf16>
    %60 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%58, %cst_33 : tensor<12x9x9xf16>, tensor<12x9x9xf16>) outs(%59 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.mulf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<12x9x9xf16>
    %expanded_34 = tensor.expand_shape %cst_27 [[0, 1], [2]] : tensor<9x9xf16> into tensor<1x9x9xf16>
    %61 = tensor.empty() : tensor<12x9x9xf16>
    %62 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %expanded_34 : tensor<12x9x9xf16>, tensor<1x9x9xf16>) outs(%61 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<12x9x9xf16>
    %63 = tensor.empty() : tensor<12x9x1xf16>
    %cst_35 = arith.constant -6.550400e+04 : f16
    %64 = linalg.fill ins(%cst_35 : f16) outs(%63 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced_36 = linalg.reduce ins(%62 : tensor<12x9x9xf16>) outs(%64 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %188 = arith.maximumf %in, %init : f16
    //     linalg.yield %188 : f16
    //   }
    // %expanded_37 = tensor.expand_shape %reduced_36 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_37 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<12x9x9xf16>) outs(%64 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %65 = tensor.empty() : tensor<12x9x9xf16>
    %66 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %expanded_37 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%65 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.subf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<12x9x9xf16>
    // %67 = math.exp %66 : tensor<12x9x9xf16>
    %68 = tensor.empty() : tensor<12x9x1xf16>
    %cst_38 = arith.constant 0.000000e+00 : f16
    %69 = linalg.fill ins(%cst_38 : f16) outs(%68 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced_39 = linalg.reduce ins(%67 : tensor<12x9x9xf16>) outs(%69 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %188 = arith.addf %in, %init : f16
    //     linalg.yield %188 : f16
    //   }
    // %expanded_40 = tensor.expand_shape %reduced_39 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_40 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%66 : tensor<12x9x9xf16>) outs(%69 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %70 = tensor.empty() : tensor<12x9x1xf16>
    %71 = linalg.generic {indexing_maps = [#map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_40 : tensor<12x9x1xf16>) outs(%70 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %188 = math.log %in : f16
      linalg.yield %188 : f16
    } -> tensor<12x9x1xf16>
    %72 = tensor.empty() : tensor<12x9x1xf16>
    %73 = linalg.generic {indexing_maps = [#map11, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37, %71 : tensor<12x9x1xf16>, tensor<12x9x1xf16>) outs(%72 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<12x9x1xf16>
    %74 = tensor.empty() : tensor<12x9x9xf16>
    %75 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %73 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%74 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.subf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<12x9x9xf16>
    // %76 = math.exp %75 : tensor<12x9x9xf16>
    %collapsed_41 = tensor.collapse_shape %73 [[0], [1, 2]] : tensor<12x9x1xf16> into tensor<12x9xf16>
    %expanded_42 = tensor.expand_shape %collapsed_41 [[0, 1], [2]] : tensor<12x9xf16> into tensor<1x12x9xf16>
    %collapsed_43 = tensor.collapse_shape %53 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %cst_44 = arith.constant 0.000000e+00 : f16
    %77 = tensor.empty() : tensor<12x9x64xf16>
    %78 = linalg.fill ins(%cst_44 : f16) outs(%77 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %79 = linalg.batch_matmul ins(%75, %collapsed_43 : tensor<12x9x9xf16>, tensor<12x9x64xf16>) outs(%78 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %expanded_45 = tensor.expand_shape %79 [[0, 1], [2], [3]] : tensor<12x9x64xf16> into tensor<1x12x9x64xf16>
    %cst_46 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %80 = tensor.empty() : tensor<1x9x12x64xf16>
    %81 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_45 : tensor<1x12x9x64xf16>) outs(%80 : tensor<1x9x12x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x9x12x64xf16>
    %collapsed_47 = tensor.collapse_shape %81 [[0], [1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<1x9x768xf16>
    %collapsed_48 = tensor.collapse_shape %81 [[0, 1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<9x768xf16>
    %82 = tensor.empty() : tensor<768x768xi8>
    %83 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<768x384xi8>) outs(%82 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.shrui %in, %c4_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<768x768xi8>
    %84 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<768x384xi8>) outs(%83 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.andi %in, %c15_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<768x768xi8>
    %cst_49 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %85 = tensor.empty() : tensor<768x6x2xf16>
    %86 = linalg.generic {indexing_maps = [#map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7 : tensor<6x768x2xf16>) outs(%85 : tensor<768x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x6x2xf16>
    %extracted_slice_50 = tensor.extract_slice %86[0, 0, 0] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %extracted_slice_51 = tensor.extract_slice %86[0, 0, 1] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %87 = tensor.empty() : tensor<768x768xf16>
    %88 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%84 : tensor<768x768xi8>) outs(%87 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %188 = arith.sitofp %in : i8 to f16
      linalg.yield %188 : f16
    } -> tensor<768x768xf16>
    %expanded_52 = tensor.expand_shape %88 [[0], [1, 2]] : tensor<768x768xf16> into tensor<768x6x128xf16>
    %89 = tensor.empty() : tensor<768x6x128xf16>
    %90 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_52, %extracted_slice_50 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%89 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.mulf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<768x6x128xf16>
    %91 = tensor.empty() : tensor<768x6x128xf16>
    %92 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %extracted_slice_51 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%91 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<768x6x128xf16>
    %collapsed_53 = tensor.collapse_shape %92 [[0], [1, 2]] : tensor<768x6x128xf16> into tensor<768x768xf16>
    %cst_54 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %93 = tensor.empty() : tensor<768x768xf16>
    %94 = linalg.generic {indexing_maps = [#map12, #map7], iterator_types = ["parallel", "parallel"]} ins(%collapsed_53 : tensor<768x768xf16>) outs(%93 : tensor<768x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x768xf16>
    // %cst_55 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %202 = tensor.empty() : tensor<9x768xf16>
    %203 = arith.constant 0.000000e+00 : f16
    %cst_55 = linalg.fill ins(%203 : f16) outs(%202 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %95 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_48, %94 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_55 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_56 = tensor.expand_shape %arg8 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %96 = tensor.empty() : tensor<9x768xf16>
    %97 = linalg.generic {indexing_maps = [#map7, #map13, #map7], iterator_types = ["parallel", "parallel"]} ins(%95, %expanded_56 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%96 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<9x768xf16>
    %expanded_57 = tensor.expand_shape %97 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %98 = tensor.empty() : tensor<1x9x768xf16>
    %99 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_57 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%98 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<1x9x768xf16>
    %100 = tensor.empty() : tensor<1x9x768xf32>
    %101 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99 : tensor<1x9x768xf16>) outs(%100 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %188 = arith.extf %in : f16 to f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %102 = tensor.empty() : tensor<1x9x1xf32>
    %cst_58 = arith.constant 0.000000e+00 : f32
    %103 = linalg.fill ins(%cst_58 : f32) outs(%102 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_59 = linalg.reduce ins(%101 : tensor<1x9x768xf32>) outs(%103 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %188 = arith.addf %in, %init : f32
    //     linalg.yield %188 : f32
    //   }
    // %expanded_60 = tensor.expand_shape %reduced_59 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_60 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%101 : tensor<1x9x768xf32>) outs(%103 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_61 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %104 = tensor.empty() : tensor<1xf32>
    %105 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_61 : tensor<1xf32>) outs(%104 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_96 = arith.constant 1.000000e+00 : f32
      %188 = arith.divf %cst_96, %in : f32
      linalg.yield %188 : f32
    } -> tensor<1xf32>
    %expanded_62 = tensor.expand_shape %105 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %106 = tensor.empty() : tensor<1x9x1xf32>
    %107 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_62, %expanded_60 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%106 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %108 = tensor.empty() : tensor<1x9x768xf32>
    %109 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %107 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%108 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.subf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %110 = tensor.empty() : tensor<1x9x768xf32>
    %111 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%109, %109 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%110 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %112 = tensor.empty() : tensor<1x9x1xf32>
    %cst_63 = arith.constant 0.000000e+00 : f32
    %113 = linalg.fill ins(%cst_63 : f32) outs(%112 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_64 = linalg.reduce ins(%111 : tensor<1x9x768xf32>) outs(%113 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %188 = arith.addf %in, %init : f32
    //     linalg.yield %188 : f32
    //   }
    // %expanded_65 = tensor.expand_shape %reduced_64 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_65 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111 : tensor<1x9x768xf32>) outs(%113 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_66 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %114 = tensor.empty() : tensor<1xf32>
    %115 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel"]} ins(%cst_66 : tensor<1xf32>) outs(%114 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_96 = arith.constant 1.000000e+00 : f32
      %188 = arith.divf %cst_96, %in : f32
      linalg.yield %188 : f32
    } -> tensor<1xf32>
    %expanded_67 = tensor.expand_shape %115 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %116 = tensor.empty() : tensor<1x9x1xf32>
    %117 = linalg.generic {indexing_maps = [#map4, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_67, %expanded_65 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%116 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %cst_68 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %118 = tensor.empty() : tensor<1x9x1xf32>
    %119 = linalg.generic {indexing_maps = [#map5, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %cst_68 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%118 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.addf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %120 = tensor.empty() : tensor<1x9x1xf32>
    %121 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119 : tensor<1x9x1xf32>) outs(%120 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %188 = math.rsqrt %in : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x1xf32>
    %122 = tensor.empty() : tensor<1x9x768xf32>
    %123 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %107 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%122 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.subf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %124 = tensor.empty() : tensor<1x9x768xf32>
    %125 = linalg.generic {indexing_maps = [#map, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %121 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%124 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %expanded_69 = tensor.expand_shape %arg10 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %126 = tensor.empty() : tensor<1x9x768xf32>
    %127 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %expanded_69 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%126 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %expanded_70 = tensor.expand_shape %arg11 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %128 = tensor.empty() : tensor<1x9x768xf32>
    %129 = linalg.generic {indexing_maps = [#map, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127, %expanded_70 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%128 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.addf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<1x9x768xf32>
    %130 = tensor.empty() : tensor<1x9x768xf16>
    %131 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129 : tensor<1x9x768xf32>) outs(%130 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %188 = arith.truncf %in : f32 to f16
      linalg.yield %188 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_71 = tensor.collapse_shape %131 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %132 = tensor.empty() : tensor<3072x768xi8>
    %133 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg12 : tensor<3072x384xi8>) outs(%132 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.shrui %in, %c4_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<3072x768xi8>
    %134 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel"]} ins(%arg12 : tensor<3072x384xi8>) outs(%133 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.andi %in, %c15_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<3072x768xi8>
    %cst_72 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %135 = tensor.empty() : tensor<3072x6x2xf16>
    %136 = linalg.generic {indexing_maps = [#map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg13 : tensor<6x3072x2xf16>) outs(%135 : tensor<3072x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<3072x6x2xf16>
    %extracted_slice_73 = tensor.extract_slice %136[0, 0, 0] [3072, 6, 1] [1, 1, 1] : tensor<3072x6x2xf16> to tensor<3072x6x1xf16>
    %extracted_slice_74 = tensor.extract_slice %136[0, 0, 1] [3072, 6, 1] [1, 1, 1] : tensor<3072x6x2xf16> to tensor<3072x6x1xf16>
    %137 = tensor.empty() : tensor<3072x768xf16>
    %138 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%134 : tensor<3072x768xi8>) outs(%137 : tensor<3072x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %188 = arith.sitofp %in : i8 to f16
      linalg.yield %188 : f16
    } -> tensor<3072x768xf16>
    %expanded_75 = tensor.expand_shape %138 [[0], [1, 2]] : tensor<3072x768xf16> into tensor<3072x6x128xf16>
    %139 = tensor.empty() : tensor<3072x6x128xf16>
    %140 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_75, %extracted_slice_73 : tensor<3072x6x128xf16>, tensor<3072x6x1xf16>) outs(%139 : tensor<3072x6x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.mulf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<3072x6x128xf16>
    %141 = tensor.empty() : tensor<3072x6x128xf16>
    %142 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %extracted_slice_74 : tensor<3072x6x128xf16>, tensor<3072x6x1xf16>) outs(%141 : tensor<3072x6x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<3072x6x128xf16>
    %collapsed_76 = tensor.collapse_shape %142 [[0], [1, 2]] : tensor<3072x6x128xf16> into tensor<3072x768xf16>
    %cst_77 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %143 = tensor.empty() : tensor<768x3072xf16>
    %144 = linalg.generic {indexing_maps = [#map12, #map7], iterator_types = ["parallel", "parallel"]} ins(%collapsed_76 : tensor<3072x768xf16>) outs(%143 : tensor<768x3072xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x3072xf16>
    // %cst_78 = arith.constant dense<0.000000e+00> : tensor<9x3072xf16>
    %204 = tensor.empty() : tensor<9x3072xf16>
    %205 = arith.constant 0.000000e+00 : f16
    %cst_78 = linalg.fill ins(%205 : f16) outs(%204 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %145 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_71, %144 : tensor<9x768xf16>, tensor<768x3072xf16>) outs(%cst_78 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %expanded_79 = tensor.expand_shape %arg14 [[0, 1]] : tensor<3072xf16> into tensor<1x3072xf16>
    %146 = tensor.empty() : tensor<9x3072xf16>
    %147 = linalg.generic {indexing_maps = [#map7, #map13, #map7], iterator_types = ["parallel", "parallel"]} ins(%145, %expanded_79 : tensor<9x3072xf16>, tensor<1x3072xf16>) outs(%146 : tensor<9x3072xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<9x3072xf16>
    %148 = tensor.empty() : tensor<9x3072xf32>
    %149 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%147 : tensor<9x3072xf16>) outs(%148 : tensor<9x3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %188 = arith.extf %in : f16 to f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %150 = tensor.empty() : tensor<9x3072xf32>
    %151 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%149, %149 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%150 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %152 = tensor.empty() : tensor<9x3072xf32>
    %153 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%151, %149 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%152 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %cst_80 = arith.constant dense<4.471500e-02> : tensor<1xf32>
    %cst_81 = arith.constant dense<4.471500e-02> : tensor<1x1xf32>
    %154 = tensor.empty() : tensor<9x3072xf32>
    %155 = linalg.generic {indexing_maps = [#map7, #map17, #map7], iterator_types = ["parallel", "parallel"]} ins(%153, %cst_81 : tensor<9x3072xf32>, tensor<1x1xf32>) outs(%154 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %156 = tensor.empty() : tensor<9x3072xf32>
    %157 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%149, %155 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%156 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.addf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %cst_82 = arith.constant dense<0.797884583> : tensor<1xf32>
    %cst_83 = arith.constant dense<0.797884583> : tensor<1x1xf32>
    %158 = tensor.empty() : tensor<9x3072xf32>
    %159 = linalg.generic {indexing_maps = [#map7, #map17, #map7], iterator_types = ["parallel", "parallel"]} ins(%157, %cst_83 : tensor<9x3072xf32>, tensor<1x1xf32>) outs(%158 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %cst_84 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %cst_85 = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
    %160 = tensor.empty() : tensor<9x3072xf32>
    %161 = linalg.generic {indexing_maps = [#map7, #map17, #map7], iterator_types = ["parallel", "parallel"]} ins(%149, %cst_85 : tensor<9x3072xf32>, tensor<1x1xf32>) outs(%160 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %162 = tensor.empty() : tensor<9x3072xf32>
    %163 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%159 : tensor<9x3072xf32>) outs(%162 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %188 = math.tanh %in : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %cst_86 = arith.constant dense<1.000000e+00> : tensor<9x3072xf32>
    %164 = tensor.empty() : tensor<9x3072xf32>
    %165 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%163, %cst_86 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%164 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.addf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %166 = tensor.empty() : tensor<9x3072xf32>
    %167 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%161, %165 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%166 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_96: f32, %out: f32):
      %188 = arith.mulf %in, %in_96 : f32
      linalg.yield %188 : f32
    } -> tensor<9x3072xf32>
    %168 = tensor.empty() : tensor<9x3072xf16>
    %169 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%167 : tensor<9x3072xf32>) outs(%168 : tensor<9x3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %188 = arith.truncf %in : f32 to f16
      linalg.yield %188 : f16
    } -> tensor<9x3072xf16>
    %170 = tensor.empty() : tensor<768x3072xi8>
    %171 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel"]} ins(%arg15 : tensor<768x1536xi8>) outs(%170 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.shrui %in, %c4_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<768x3072xi8>
    %172 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel"]} ins(%arg15 : tensor<768x1536xi8>) outs(%171 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %188 = arith.andi %in, %c15_i8 : i8
      %189 = arith.subi %188, %c8_i8 : i8
      linalg.yield %189 : i8
    } -> tensor<768x3072xi8>
    %cst_87 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %173 = tensor.empty() : tensor<768x24x2xf16>
    %174 = linalg.generic {indexing_maps = [#map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg16 : tensor<24x768x2xf16>) outs(%173 : tensor<768x24x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x24x2xf16>
    %extracted_slice_88 = tensor.extract_slice %174[0, 0, 0] [768, 24, 1] [1, 1, 1] : tensor<768x24x2xf16> to tensor<768x24x1xf16>
    %extracted_slice_89 = tensor.extract_slice %174[0, 0, 1] [768, 24, 1] [1, 1, 1] : tensor<768x24x2xf16> to tensor<768x24x1xf16>
    %175 = tensor.empty() : tensor<768x3072xf16>
    %176 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%172 : tensor<768x3072xi8>) outs(%175 : tensor<768x3072xf16>) {
    ^bb0(%in: i8, %out: f16):
      %188 = arith.sitofp %in : i8 to f16
      linalg.yield %188 : f16
    } -> tensor<768x3072xf16>
    %expanded_90 = tensor.expand_shape %176 [[0], [1, 2]] : tensor<768x3072xf16> into tensor<768x24x128xf16>
    %177 = tensor.empty() : tensor<768x24x128xf16>
    %178 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_90, %extracted_slice_88 : tensor<768x24x128xf16>, tensor<768x24x1xf16>) outs(%177 : tensor<768x24x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.mulf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<768x24x128xf16>
    %179 = tensor.empty() : tensor<768x24x128xf16>
    %180 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178, %extracted_slice_89 : tensor<768x24x128xf16>, tensor<768x24x1xf16>) outs(%179 : tensor<768x24x128xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<768x24x128xf16>
    %collapsed_91 = tensor.collapse_shape %180 [[0], [1, 2]] : tensor<768x24x128xf16> into tensor<768x3072xf16>
    %cst_92 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %181 = tensor.empty() : tensor<3072x768xf16>
    %182 = linalg.generic {indexing_maps = [#map12, #map7], iterator_types = ["parallel", "parallel"]} ins(%collapsed_91 : tensor<768x3072xf16>) outs(%181 : tensor<3072x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<3072x768xf16>
    // %cst_93 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %206 = tensor.empty() : tensor<9x768xf16>
    %207 = arith.constant 0.000000e+00 : f16
    %cst_93 = linalg.fill ins(%207 : f16) outs(%206 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %183 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%169, %182 : tensor<9x3072xf16>, tensor<3072x768xf16>) outs(%cst_93 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_94 = tensor.expand_shape %arg17 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %184 = tensor.empty() : tensor<9x768xf16>
    %185 = linalg.generic {indexing_maps = [#map7, #map13, #map7], iterator_types = ["parallel", "parallel"]} ins(%183, %expanded_94 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%184 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<9x768xf16>
    %expanded_95 = tensor.expand_shape %185 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %186 = tensor.empty() : tensor<1x9x768xf16>
    %187 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99, %expanded_95 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%186 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_96: f16, %out: f16):
      %188 = arith.addf %in, %in_96 : f16
      linalg.yield %188 : f16
    } -> tensor<1x9x768xf16>
    return %187 : tensor<1x9x768xf16>
  }
}

