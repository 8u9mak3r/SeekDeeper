#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 * 2)>
#map2 = affine_map<(d0, d1) -> (d0, d1 * 2 + 1)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map6 = affine_map<(d0, d1) -> (d1, d0)>
#map7 = affine_map<(d0, d1) -> (0, d1)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map12 = affine_map<(d0) -> (0)>
#map13 = affine_map<(d0) -> (d0)>
#map14 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map15 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map16 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map17 = affine_map<(d0, d1) -> (0, 0)>
module {
  func.func @forward(%arg0: tensor<1x9x768xf16>, %arg1: tensor<768x384xi8>, %arg2: tensor<6x768x2xf16>, %arg3: tensor<768xf16>, %arg4: tensor<1x9x768xf16>, %arg5: tensor<768x384xi8>, %arg6: tensor<6x768x2xf16>, %arg7: tensor<768xf16>, %arg8: tensor<1x9x768xf16>, %arg9: tensor<768x384xi8>, %arg10: tensor<6x768x2xf16>, %arg11: tensor<768xf16>, %arg12: tensor<768x384xi8>, %arg13: tensor<6x768x2xf16>, %arg14: tensor<768xf16>, %arg15: tensor<1x9x768xf16>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<3072x384xi8>, %arg19: tensor<6x3072x2xf16>, %arg20: tensor<3072xf16>, %arg21: tensor<768x1536xi8>, %arg22: tensor<24x768x2xf16>, %arg23: tensor<768xf16>, %arg24: tensor<768xf32>, %arg25: tensor<768xf32>) -> tensor<1x9x768xf16> {
    %collapsed = tensor.collapse_shape %arg15 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %0 = tensor.empty() : tensor<768x768xi8>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<768x384xi8>) outs(%0 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.shrui %in, %c4_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %2 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<768x384xi8>) outs(%1 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.andi %in, %c15_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %cst = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %3 = tensor.empty() : tensor<768x6x2xf16>
    %4 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<6x768x2xf16>) outs(%3 : tensor<768x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x6x2xf16>
    %extracted_slice = tensor.extract_slice %4[0, 0, 0] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %extracted_slice_0 = tensor.extract_slice %4[0, 0, 1] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %5 = tensor.empty() : tensor<768x768xf16>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<768x768xi8>) outs(%5 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %211 = arith.sitofp %in : i8 to f16
      linalg.yield %211 : f16
    } -> tensor<768x768xf16>
    %expanded = tensor.expand_shape %6 [[0], [1, 2]] : tensor<768x768xf16> into tensor<768x6x128xf16>
    %7 = tensor.empty() : tensor<768x6x128xf16>
    %8 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %extracted_slice : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%7 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %9 = tensor.empty() : tensor<768x6x128xf16>
    %10 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %extracted_slice_0 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%9 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %collapsed_1 = tensor.collapse_shape %10 [[0], [1, 2]] : tensor<768x6x128xf16> into tensor<768x768xf16>
    %cst_2 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %11 = tensor.empty() : tensor<768x768xf16>
    %12 = linalg.generic {indexing_maps = [#map6, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_1 : tensor<768x768xf16>) outs(%11 : tensor<768x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x768xf16>
    // %cst_3 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %300 = tensor.empty() : tensor<9x768xf16>
    %cst_301 = arith.constant 0.000000e+00 : f16
    %cst_3 = linalg.fill ins(%cst_301 : f16) outs(%300 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %13 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed, %12 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_3 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_4 = tensor.expand_shape %arg3 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %14 = tensor.empty() : tensor<9x768xf16>
    %15 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %expanded_4 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%14 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<9x768xf16>
    %collapsed_5 = tensor.collapse_shape %arg15 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %16 = tensor.empty() : tensor<768x768xi8>
    %17 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg5 : tensor<768x384xi8>) outs(%16 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.shrui %in, %c4_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %18 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg5 : tensor<768x384xi8>) outs(%17 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.andi %in, %c15_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %cst_6 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %19 = tensor.empty() : tensor<768x6x2xf16>
    %20 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg6 : tensor<6x768x2xf16>) outs(%19 : tensor<768x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x6x2xf16>
    %extracted_slice_7 = tensor.extract_slice %20[0, 0, 0] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %extracted_slice_8 = tensor.extract_slice %20[0, 0, 1] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %21 = tensor.empty() : tensor<768x768xf16>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<768x768xi8>) outs(%21 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %211 = arith.sitofp %in : i8 to f16
      linalg.yield %211 : f16
    } -> tensor<768x768xf16>
    %expanded_9 = tensor.expand_shape %22 [[0], [1, 2]] : tensor<768x768xf16> into tensor<768x6x128xf16>
    %23 = tensor.empty() : tensor<768x6x128xf16>
    %24 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_9, %extracted_slice_7 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%23 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %25 = tensor.empty() : tensor<768x6x128xf16>
    %26 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %extracted_slice_8 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%25 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %collapsed_10 = tensor.collapse_shape %26 [[0], [1, 2]] : tensor<768x6x128xf16> into tensor<768x768xf16>
    %cst_11 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %27 = tensor.empty() : tensor<768x768xf16>
    %28 = linalg.generic {indexing_maps = [#map6, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_10 : tensor<768x768xf16>) outs(%27 : tensor<768x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x768xf16>
    // %cst_12 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %302 = tensor.empty() : tensor<9x768xf16>
    %cst_303 = arith.constant 0.000000e+00 : f16
    %cst_12 = linalg.fill ins(%cst_303 : f16) outs(%302 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %29 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_5, %28 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_12 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_13 = tensor.expand_shape %arg7 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %30 = tensor.empty() : tensor<9x768xf16>
    %31 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel"]} ins(%29, %expanded_13 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%30 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<9x768xf16>
    %collapsed_14 = tensor.collapse_shape %arg15 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %32 = tensor.empty() : tensor<768x768xi8>
    %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg9 : tensor<768x384xi8>) outs(%32 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.shrui %in, %c4_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %34 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg9 : tensor<768x384xi8>) outs(%33 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.andi %in, %c15_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %cst_15 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %35 = tensor.empty() : tensor<768x6x2xf16>
    %36 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10 : tensor<6x768x2xf16>) outs(%35 : tensor<768x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x6x2xf16>
    %extracted_slice_16 = tensor.extract_slice %36[0, 0, 0] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %extracted_slice_17 = tensor.extract_slice %36[0, 0, 1] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %37 = tensor.empty() : tensor<768x768xf16>
    %38 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<768x768xi8>) outs(%37 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %211 = arith.sitofp %in : i8 to f16
      linalg.yield %211 : f16
    } -> tensor<768x768xf16>
    %expanded_18 = tensor.expand_shape %38 [[0], [1, 2]] : tensor<768x768xf16> into tensor<768x6x128xf16>
    %39 = tensor.empty() : tensor<768x6x128xf16>
    %40 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_18, %extracted_slice_16 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%39 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %41 = tensor.empty() : tensor<768x6x128xf16>
    %42 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %extracted_slice_17 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%41 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %collapsed_19 = tensor.collapse_shape %42 [[0], [1, 2]] : tensor<768x6x128xf16> into tensor<768x768xf16>
    %cst_20 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %43 = tensor.empty() : tensor<768x768xf16>
    %44 = linalg.generic {indexing_maps = [#map6, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_19 : tensor<768x768xf16>) outs(%43 : tensor<768x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x768xf16>
    // %cst_21 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %304 = tensor.empty() : tensor<9x768xf16>
    %cst_305 = arith.constant 0.000000e+00 : f16
    %cst_21 = linalg.fill ins(%cst_305 : f16) outs(%304 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_14, %44 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_21 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_22 = tensor.expand_shape %arg11 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %46 = tensor.empty() : tensor<9x768xf16>
    %47 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel"]} ins(%45, %expanded_22 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%46 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<9x768xf16>
    %expanded_23 = tensor.expand_shape %15 [[0, 1], [2, 3]] : tensor<9x768xf16> into tensor<1x9x12x64xf16>
    %cst_24 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %48 = tensor.empty() : tensor<1x12x9x64xf16>
    %49 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_23 : tensor<1x9x12x64xf16>) outs(%48 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %expanded_25 = tensor.expand_shape %31 [[0, 1], [2, 3]] : tensor<9x768xf16> into tensor<1x9x12x64xf16>
    %cst_26 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %50 = tensor.empty() : tensor<1x12x9x64xf16>
    %51 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_25 : tensor<1x9x12x64xf16>) outs(%50 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %expanded_27 = tensor.expand_shape %47 [[0, 1], [2, 3]] : tensor<9x768xf16> into tensor<1x9x12x64xf16>
    %cst_28 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %52 = tensor.empty() : tensor<1x12x9x64xf16>
    %53 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_27 : tensor<1x9x12x64xf16>) outs(%52 : tensor<1x12x9x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x9x64xf16>
    %cst_29 = arith.constant 0.000000e+00 : f16
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
    %cst_31 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %54 = tensor.empty() : tensor<1x12x64x9xf16>
    %55 = linalg.generic {indexing_maps = [#map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51 : tensor<1x12x9x64xf16>) outs(%54 : tensor<1x12x64x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x12x64x9xf16>
    %collapsed_32 = tensor.collapse_shape %49 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %collapsed_33 = tensor.collapse_shape %55 [[0, 1], [2], [3]] : tensor<1x12x64x9xf16> into tensor<12x64x9xf16>
    %cst_34 = arith.constant 0.000000e+00 : f16
    %56 = tensor.empty() : tensor<12x9x9xf16>
    %57 = linalg.fill ins(%cst_34 : f16) outs(%56 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %58 = linalg.batch_matmul ins(%collapsed_32, %collapsed_33 : tensor<12x9x64xf16>, tensor<12x64x9xf16>) outs(%57 : tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %cst_35 = arith.constant 1.250000e-01 : f16
    %cst_36 = arith.constant dense<1.250000e-01> : tensor<12x9x9xf16>
    %59 = tensor.empty() : tensor<12x9x9xf16>
    %60 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%58, %cst_36 : tensor<12x9x9xf16>, tensor<12x9x9xf16>) outs(%59 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<12x9x9xf16>
    %expanded_37 = tensor.expand_shape %cst_30 [[0, 1], [2]] : tensor<9x9xf16> into tensor<1x9x9xf16>
    %61 = tensor.empty() : tensor<12x9x9xf16>
    %62 = linalg.generic {indexing_maps = [#map4, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %expanded_37 : tensor<12x9x9xf16>, tensor<1x9x9xf16>) outs(%61 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<12x9x9xf16>
    %63 = tensor.empty() : tensor<12x9x1xf16>
    %cst_38 = arith.constant -6.550400e+04 : f16
    %64 = linalg.fill ins(%cst_38 : f16) outs(%63 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced = linalg.reduce ins(%62 : tensor<12x9x9xf16>) outs(%64 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %211 = arith.maximumf %in, %init : f16
    //     linalg.yield %211 : f16
    //   }
    // %expanded_39 = tensor.expand_shape %reduced [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_39 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<12x9x9xf16>) outs(%64 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %65 = tensor.empty() : tensor<12x9x9xf16>
    %66 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %expanded_39 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%65 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.subf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<12x9x9xf16>
    // %67 = math.exp %66 : tensor<12x9x9xf16>
    %68 = tensor.empty() : tensor<12x9x1xf16>
    %cst_40 = arith.constant 0.000000e+00 : f16
    %69 = linalg.fill ins(%cst_40 : f16) outs(%68 : tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    // %reduced_41 = linalg.reduce ins(%67 : tensor<12x9x9xf16>) outs(%69 : tensor<12x9xf16>) dimensions = [2] 
    //   (%in: f16, %init: f16) {
    //     %211 = arith.addf %in, %init : f16
    //     linalg.yield %211 : f16
    //   }
    // %expanded_42 = tensor.expand_shape %reduced_41 [[0], [1, 2]] : tensor<12x9xf16> into tensor<12x9x1xf16>
    %expanded_42 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%66 : tensor<12x9x9xf16>) outs(%69 : tensor<12x9x1xf16>) {
      ^bb0(%in: f16, %init: f16):
        %864 = arith.addf %in, %init : f16
        linalg.yield %864 : f16
    } -> tensor<12x9x1xf16>
    %70 = tensor.empty() : tensor<12x9x1xf16>
    %71 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_42 : tensor<12x9x1xf16>) outs(%70 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %211 = math.log %in : f16
      linalg.yield %211 : f16
    } -> tensor<12x9x1xf16>
    %72 = tensor.empty() : tensor<12x9x1xf16>
    %73 = linalg.generic {indexing_maps = [#map5, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_39, %71 : tensor<12x9x1xf16>, tensor<12x9x1xf16>) outs(%72 : tensor<12x9x1xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<12x9x1xf16>
    %74 = tensor.empty() : tensor<12x9x9xf16>
    %75 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %73 : tensor<12x9x9xf16>, tensor<12x9x1xf16>) outs(%74 : tensor<12x9x9xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.subf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<12x9x9xf16>
    // %76 = math.exp %75 : tensor<12x9x9xf16>
    %collapsed_43 = tensor.collapse_shape %73 [[0], [1, 2]] : tensor<12x9x1xf16> into tensor<12x9xf16>
    %expanded_44 = tensor.expand_shape %collapsed_43 [[0, 1], [2]] : tensor<12x9xf16> into tensor<1x12x9xf16>
    %collapsed_45 = tensor.collapse_shape %53 [[0, 1], [2], [3]] : tensor<1x12x9x64xf16> into tensor<12x9x64xf16>
    %cst_46 = arith.constant 0.000000e+00 : f16
    %77 = tensor.empty() : tensor<12x9x64xf16>
    %78 = linalg.fill ins(%cst_46 : f16) outs(%77 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %79 = linalg.batch_matmul ins(%75, %collapsed_45 : tensor<12x9x9xf16>, tensor<12x9x64xf16>) outs(%78 : tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %expanded_47 = tensor.expand_shape %79 [[0, 1], [2], [3]] : tensor<12x9x64xf16> into tensor<1x12x9x64xf16>
    %cst_48 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %80 = tensor.empty() : tensor<1x9x12x64xf16>
    %81 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_47 : tensor<1x12x9x64xf16>) outs(%80 : tensor<1x9x12x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x9x12x64xf16>
    %collapsed_49 = tensor.collapse_shape %81 [[0], [1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<1x9x768xf16>
    %collapsed_50 = tensor.collapse_shape %81 [[0, 1], [2, 3]] : tensor<1x9x12x64xf16> into tensor<9x768xf16>
    %82 = tensor.empty() : tensor<768x768xi8>
    %83 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg12 : tensor<768x384xi8>) outs(%82 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.shrui %in, %c4_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %84 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg12 : tensor<768x384xi8>) outs(%83 : tensor<768x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.andi %in, %c15_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x768xi8>
    %cst_51 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %85 = tensor.empty() : tensor<768x6x2xf16>
    %86 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg13 : tensor<6x768x2xf16>) outs(%85 : tensor<768x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x6x2xf16>
    %extracted_slice_52 = tensor.extract_slice %86[0, 0, 0] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %extracted_slice_53 = tensor.extract_slice %86[0, 0, 1] [768, 6, 1] [1, 1, 1] : tensor<768x6x2xf16> to tensor<768x6x1xf16>
    %87 = tensor.empty() : tensor<768x768xf16>
    %88 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%84 : tensor<768x768xi8>) outs(%87 : tensor<768x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %211 = arith.sitofp %in : i8 to f16
      linalg.yield %211 : f16
    } -> tensor<768x768xf16>
    %expanded_54 = tensor.expand_shape %88 [[0], [1, 2]] : tensor<768x768xf16> into tensor<768x6x128xf16>
    %89 = tensor.empty() : tensor<768x6x128xf16>
    %90 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_54, %extracted_slice_52 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%89 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %91 = tensor.empty() : tensor<768x6x128xf16>
    %92 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %extracted_slice_53 : tensor<768x6x128xf16>, tensor<768x6x1xf16>) outs(%91 : tensor<768x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x6x128xf16>
    %collapsed_55 = tensor.collapse_shape %92 [[0], [1, 2]] : tensor<768x6x128xf16> into tensor<768x768xf16>
    %cst_56 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %93 = tensor.empty() : tensor<768x768xf16>
    %94 = linalg.generic {indexing_maps = [#map6, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_55 : tensor<768x768xf16>) outs(%93 : tensor<768x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x768xf16>
    // %cst_57 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %306 = tensor.empty() : tensor<9x768xf16>
    %cst_307 = arith.constant 0.000000e+00 : f16
    %cst_57 = linalg.fill ins(%cst_307 : f16) outs(%306 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %95 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_50, %94 : tensor<9x768xf16>, tensor<768x768xf16>) outs(%cst_57 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_58 = tensor.expand_shape %arg14 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %96 = tensor.empty() : tensor<9x768xf16>
    %97 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel"]} ins(%95, %expanded_58 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%96 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<9x768xf16>
    %expanded_59 = tensor.expand_shape %97 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %98 = tensor.empty() : tensor<1x9x768xf16>
    %99 = linalg.generic {indexing_maps = [#map11, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg15, %expanded_59 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%98 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<1x9x768xf16>
    %100 = tensor.empty() : tensor<1x9x768xf32>
    %101 = linalg.generic {indexing_maps = [#map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99 : tensor<1x9x768xf16>) outs(%100 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %211 = arith.extf %in : f16 to f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %102 = tensor.empty() : tensor<1x9x1xf32>
    %cst_60 = arith.constant 0.000000e+00 : f32
    %103 = linalg.fill ins(%cst_60 : f32) outs(%102 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_61 = linalg.reduce ins(%101 : tensor<1x9x768xf32>) outs(%103 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %211 = arith.addf %in, %init : f32
    //     linalg.yield %211 : f32
    //   }
    // %expanded_62 = tensor.expand_shape %reduced_61 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_62 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%101 : tensor<1x9x768xf32>) outs(%103 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_63 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %104 = tensor.empty() : tensor<1xf32>
    %105 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel"]} ins(%cst_63 : tensor<1xf32>) outs(%104 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_109 = arith.constant 1.000000e+00 : f32
      %211 = arith.divf %cst_109, %in : f32
      linalg.yield %211 : f32
    } -> tensor<1xf32>
    %expanded_64 = tensor.expand_shape %105 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %106 = tensor.empty() : tensor<1x9x1xf32>
    %107 = linalg.generic {indexing_maps = [#map14, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_64, %expanded_62 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%106 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %108 = tensor.empty() : tensor<1x9x768xf32>
    %109 = linalg.generic {indexing_maps = [#map11, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %107 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%108 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.subf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %110 = tensor.empty() : tensor<1x9x768xf32>
    %111 = linalg.generic {indexing_maps = [#map11, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%109, %109 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%110 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %112 = tensor.empty() : tensor<1x9x1xf32>
    %cst_65 = arith.constant 0.000000e+00 : f32
    %113 = linalg.fill ins(%cst_65 : f32) outs(%112 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_66 = linalg.reduce ins(%111 : tensor<1x9x768xf32>) outs(%113 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %211 = arith.addf %in, %init : f32
    //     linalg.yield %211 : f32
    //   }
    // %expanded_67 = tensor.expand_shape %reduced_66 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_67 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111 : tensor<1x9x768xf32>) outs(%113 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_68 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %114 = tensor.empty() : tensor<1xf32>
    %115 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel"]} ins(%cst_68 : tensor<1xf32>) outs(%114 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_109 = arith.constant 1.000000e+00 : f32
      %211 = arith.divf %cst_109, %in : f32
      linalg.yield %211 : f32
    } -> tensor<1xf32>
    %expanded_69 = tensor.expand_shape %115 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %116 = tensor.empty() : tensor<1x9x1xf32>
    %117 = linalg.generic {indexing_maps = [#map14, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_69, %expanded_67 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%116 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %cst_70 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %118 = tensor.empty() : tensor<1x9x1xf32>
    %119 = linalg.generic {indexing_maps = [#map15, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %cst_70 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%118 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.addf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %120 = tensor.empty() : tensor<1x9x1xf32>
    %121 = linalg.generic {indexing_maps = [#map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119 : tensor<1x9x1xf32>) outs(%120 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %211 = math.rsqrt %in : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %122 = tensor.empty() : tensor<1x9x768xf32>
    %123 = linalg.generic {indexing_maps = [#map11, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %107 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%122 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.subf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %124 = tensor.empty() : tensor<1x9x768xf32>
    %125 = linalg.generic {indexing_maps = [#map11, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %121 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%124 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %expanded_71 = tensor.expand_shape %arg16 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %126 = tensor.empty() : tensor<1x9x768xf32>
    %127 = linalg.generic {indexing_maps = [#map11, #map16, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %expanded_71 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%126 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %expanded_72 = tensor.expand_shape %arg17 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %128 = tensor.empty() : tensor<1x9x768xf32>
    %129 = linalg.generic {indexing_maps = [#map11, #map16, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127, %expanded_72 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%128 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.addf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %130 = tensor.empty() : tensor<1x9x768xf16>
    %131 = linalg.generic {indexing_maps = [#map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129 : tensor<1x9x768xf32>) outs(%130 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %211 = arith.truncf %in : f32 to f16
      linalg.yield %211 : f16
    } -> tensor<1x9x768xf16>
    %collapsed_73 = tensor.collapse_shape %131 [[0, 1], [2]] : tensor<1x9x768xf16> into tensor<9x768xf16>
    %132 = tensor.empty() : tensor<3072x768xi8>
    %133 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg18 : tensor<3072x384xi8>) outs(%132 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.shrui %in, %c4_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<3072x768xi8>
    %134 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg18 : tensor<3072x384xi8>) outs(%133 : tensor<3072x768xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.andi %in, %c15_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<3072x768xi8>
    %cst_74 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %135 = tensor.empty() : tensor<3072x6x2xf16>
    %136 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg19 : tensor<6x3072x2xf16>) outs(%135 : tensor<3072x6x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<3072x6x2xf16>
    %extracted_slice_75 = tensor.extract_slice %136[0, 0, 0] [3072, 6, 1] [1, 1, 1] : tensor<3072x6x2xf16> to tensor<3072x6x1xf16>
    %extracted_slice_76 = tensor.extract_slice %136[0, 0, 1] [3072, 6, 1] [1, 1, 1] : tensor<3072x6x2xf16> to tensor<3072x6x1xf16>
    %137 = tensor.empty() : tensor<3072x768xf16>
    %138 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%134 : tensor<3072x768xi8>) outs(%137 : tensor<3072x768xf16>) {
    ^bb0(%in: i8, %out: f16):
      %211 = arith.sitofp %in : i8 to f16
      linalg.yield %211 : f16
    } -> tensor<3072x768xf16>
    %expanded_77 = tensor.expand_shape %138 [[0], [1, 2]] : tensor<3072x768xf16> into tensor<3072x6x128xf16>
    %139 = tensor.empty() : tensor<3072x6x128xf16>
    %140 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_77, %extracted_slice_75 : tensor<3072x6x128xf16>, tensor<3072x6x1xf16>) outs(%139 : tensor<3072x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<3072x6x128xf16>
    %141 = tensor.empty() : tensor<3072x6x128xf16>
    %142 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %extracted_slice_76 : tensor<3072x6x128xf16>, tensor<3072x6x1xf16>) outs(%141 : tensor<3072x6x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<3072x6x128xf16>
    %collapsed_78 = tensor.collapse_shape %142 [[0], [1, 2]] : tensor<3072x6x128xf16> into tensor<3072x768xf16>
    %cst_79 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %143 = tensor.empty() : tensor<768x3072xf16>
    %144 = linalg.generic {indexing_maps = [#map6, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_78 : tensor<3072x768xf16>) outs(%143 : tensor<768x3072xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x3072xf16>
    // %cst_80 = arith.constant dense<0.000000e+00> : tensor<9x3072xf16>
    %308 = tensor.empty() : tensor<9x3072xf16>
    %cst_309 = arith.constant 0.000000e+00 : f16
    %cst_80 = linalg.fill ins(%cst_309 : f16) outs(%308 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %145 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_73, %144 : tensor<9x768xf16>, tensor<768x3072xf16>) outs(%cst_80 : tensor<9x3072xf16>) -> tensor<9x3072xf16>
    %expanded_81 = tensor.expand_shape %arg20 [[0, 1]] : tensor<3072xf16> into tensor<1x3072xf16>
    %146 = tensor.empty() : tensor<9x3072xf16>
    %147 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel"]} ins(%145, %expanded_81 : tensor<9x3072xf16>, tensor<1x3072xf16>) outs(%146 : tensor<9x3072xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<9x3072xf16>
    %148 = tensor.empty() : tensor<9x3072xf32>
    %149 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%147 : tensor<9x3072xf16>) outs(%148 : tensor<9x3072xf32>) {
    ^bb0(%in: f16, %out: f32):
      %211 = arith.extf %in : f16 to f32
      linalg.yield %211 : f32
    } -> tensor<9x3072xf32>
    %cst_82 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %cst_83 = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
    %150 = tensor.empty() : tensor<9x3072xf32>
    %151 = linalg.generic {indexing_maps = [#map, #map17, #map], iterator_types = ["parallel", "parallel"]} ins(%149, %cst_83 : tensor<9x3072xf32>, tensor<1x1xf32>) outs(%150 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<9x3072xf32>
    %cst_84 = arith.constant dense<0.707106769> : tensor<1xf32>
    %cst_85 = arith.constant dense<0.707106769> : tensor<1x1xf32>
    %152 = tensor.empty() : tensor<9x3072xf32>
    %153 = linalg.generic {indexing_maps = [#map, #map17, #map], iterator_types = ["parallel", "parallel"]} ins(%149, %cst_85 : tensor<9x3072xf32>, tensor<1x1xf32>) outs(%152 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<9x3072xf32>
    // %154 = math.erf %153 : tensor<9x3072xf32>
    %cst_86 = arith.constant dense<1.000000e+00> : tensor<9x3072xf32>
    %155 = tensor.empty() : tensor<9x3072xf32>
    %156 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%153, %cst_86 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%155 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.addf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<9x3072xf32>
    %157 = tensor.empty() : tensor<9x3072xf32>
    %158 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%151, %156 : tensor<9x3072xf32>, tensor<9x3072xf32>) outs(%157 : tensor<9x3072xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<9x3072xf32>
    %159 = tensor.empty() : tensor<9x3072xf16>
    %160 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%158 : tensor<9x3072xf32>) outs(%159 : tensor<9x3072xf16>) {
    ^bb0(%in: f32, %out: f16):
      %211 = arith.truncf %in : f32 to f16
      linalg.yield %211 : f16
    } -> tensor<9x3072xf16>
    %161 = tensor.empty() : tensor<768x3072xi8>
    %162 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg21 : tensor<768x1536xi8>) outs(%161 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c4_i8 = arith.constant 4 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.shrui %in, %c4_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x3072xi8>
    %163 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg21 : tensor<768x1536xi8>) outs(%162 : tensor<768x3072xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c15_i8 = arith.constant 15 : i8
      %c8_i8 = arith.constant 8 : i8
      %211 = arith.andi %in, %c15_i8 : i8
      %212 = arith.subi %211, %c8_i8 : i8
      linalg.yield %212 : i8
    } -> tensor<768x3072xi8>
    %cst_87 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %164 = tensor.empty() : tensor<768x24x2xf16>
    %165 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg22 : tensor<24x768x2xf16>) outs(%164 : tensor<768x24x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<768x24x2xf16>
    %extracted_slice_88 = tensor.extract_slice %165[0, 0, 0] [768, 24, 1] [1, 1, 1] : tensor<768x24x2xf16> to tensor<768x24x1xf16>
    %extracted_slice_89 = tensor.extract_slice %165[0, 0, 1] [768, 24, 1] [1, 1, 1] : tensor<768x24x2xf16> to tensor<768x24x1xf16>
    %166 = tensor.empty() : tensor<768x3072xf16>
    %167 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%163 : tensor<768x3072xi8>) outs(%166 : tensor<768x3072xf16>) {
    ^bb0(%in: i8, %out: f16):
      %211 = arith.sitofp %in : i8 to f16
      linalg.yield %211 : f16
    } -> tensor<768x3072xf16>
    %expanded_90 = tensor.expand_shape %167 [[0], [1, 2]] : tensor<768x3072xf16> into tensor<768x24x128xf16>
    %168 = tensor.empty() : tensor<768x24x128xf16>
    %169 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_90, %extracted_slice_88 : tensor<768x24x128xf16>, tensor<768x24x1xf16>) outs(%168 : tensor<768x24x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.mulf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x24x128xf16>
    %170 = tensor.empty() : tensor<768x24x128xf16>
    %171 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169, %extracted_slice_89 : tensor<768x24x128xf16>, tensor<768x24x1xf16>) outs(%170 : tensor<768x24x128xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<768x24x128xf16>
    %collapsed_91 = tensor.collapse_shape %171 [[0], [1, 2]] : tensor<768x24x128xf16> into tensor<768x3072xf16>
    %cst_92 = arith.constant dense<[1, 0]> : tensor<2xi32>
    %172 = tensor.empty() : tensor<3072x768xf16>
    %173 = linalg.generic {indexing_maps = [#map6, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_91 : tensor<768x3072xf16>) outs(%172 : tensor<3072x768xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<3072x768xf16>
    // %cst_93 = arith.constant dense<0.000000e+00> : tensor<9x768xf16>
    %310 = tensor.empty() : tensor<9x768xf16>
    %cst_311 = arith.constant 0.000000e+00 : f16
    %cst_93 = linalg.fill ins(%cst_311 : f16) outs(%310 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %174 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%160, %173 : tensor<9x3072xf16>, tensor<3072x768xf16>) outs(%cst_93 : tensor<9x768xf16>) -> tensor<9x768xf16>
    %expanded_94 = tensor.expand_shape %arg23 [[0, 1]] : tensor<768xf16> into tensor<1x768xf16>
    %175 = tensor.empty() : tensor<9x768xf16>
    %176 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel"]} ins(%174, %expanded_94 : tensor<9x768xf16>, tensor<1x768xf16>) outs(%175 : tensor<9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<9x768xf16>
    %expanded_95 = tensor.expand_shape %176 [[0, 1], [2]] : tensor<9x768xf16> into tensor<1x9x768xf16>
    %177 = tensor.empty() : tensor<1x9x768xf16>
    %178 = linalg.generic {indexing_maps = [#map11, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131, %expanded_95 : tensor<1x9x768xf16>, tensor<1x9x768xf16>) outs(%177 : tensor<1x9x768xf16>) {
    ^bb0(%in: f16, %in_109: f16, %out: f16):
      %211 = arith.addf %in, %in_109 : f16
      linalg.yield %211 : f16
    } -> tensor<1x9x768xf16>
    %179 = tensor.empty() : tensor<1x9x768xf32>
    %180 = linalg.generic {indexing_maps = [#map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178 : tensor<1x9x768xf16>) outs(%179 : tensor<1x9x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %211 = arith.extf %in : f16 to f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %181 = tensor.empty() : tensor<1x9x1xf32>
    %cst_96 = arith.constant 0.000000e+00 : f32
    %182 = linalg.fill ins(%cst_96 : f32) outs(%181 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_97 = linalg.reduce ins(%180 : tensor<1x9x768xf32>) outs(%182 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %211 = arith.addf %in, %init : f32
    //     linalg.yield %211 : f32
    //   }
    // %expanded_98 = tensor.expand_shape %reduced_97 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_98 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%180 : tensor<1x9x768xf32>) outs(%182 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_99 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %183 = tensor.empty() : tensor<1xf32>
    %184 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel"]} ins(%cst_99 : tensor<1xf32>) outs(%183 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_109 = arith.constant 1.000000e+00 : f32
      %211 = arith.divf %cst_109, %in : f32
      linalg.yield %211 : f32
    } -> tensor<1xf32>
    %expanded_100 = tensor.expand_shape %184 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %185 = tensor.empty() : tensor<1x9x1xf32>
    %186 = linalg.generic {indexing_maps = [#map14, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_100, %expanded_98 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%185 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %187 = tensor.empty() : tensor<1x9x768xf32>
    %188 = linalg.generic {indexing_maps = [#map11, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180, %186 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%187 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.subf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %189 = tensor.empty() : tensor<1x9x768xf32>
    %190 = linalg.generic {indexing_maps = [#map11, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %188 : tensor<1x9x768xf32>, tensor<1x9x768xf32>) outs(%189 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %191 = tensor.empty() : tensor<1x9x1xf32>
    %cst_101 = arith.constant 0.000000e+00 : f32
    %192 = linalg.fill ins(%cst_101 : f32) outs(%191 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    // %reduced_102 = linalg.reduce ins(%190 : tensor<1x9x768xf32>) outs(%192 : tensor<1x9xf32>) dimensions = [2] 
    //   (%in: f32, %init: f32) {
    //     %211 = arith.addf %in, %init : f32
    //     linalg.yield %211 : f32
    //   }
    // %expanded_103 = tensor.expand_shape %reduced_102 [[0], [1, 2]] : tensor<1x9xf32> into tensor<1x9x1xf32>
    %expanded_103 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%190 : tensor<1x9x768xf32>) outs(%192 : tensor<1x9x1xf32>) {
      ^bb0(%in: f32, %init: f32):
        %864 = arith.addf %in, %init : f32
        linalg.yield %864 : f32
    } -> tensor<1x9x1xf32>
    %cst_104 = arith.constant dense<7.680000e+02> : tensor<1xf32>
    %193 = tensor.empty() : tensor<1xf32>
    %194 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel"]} ins(%cst_104 : tensor<1xf32>) outs(%193 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_109 = arith.constant 1.000000e+00 : f32
      %211 = arith.divf %cst_109, %in : f32
      linalg.yield %211 : f32
    } -> tensor<1xf32>
    %expanded_105 = tensor.expand_shape %194 [[0, 1, 2]] : tensor<1xf32> into tensor<1x1x1xf32>
    %195 = tensor.empty() : tensor<1x9x1xf32>
    %196 = linalg.generic {indexing_maps = [#map14, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_105, %expanded_103 : tensor<1x1x1xf32>, tensor<1x9x1xf32>) outs(%195 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %cst_106 = arith.constant dense<9.99999974E-6> : tensor<1x9x1xf32>
    %197 = tensor.empty() : tensor<1x9x1xf32>
    %198 = linalg.generic {indexing_maps = [#map15, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%196, %cst_106 : tensor<1x9x1xf32>, tensor<1x9x1xf32>) outs(%197 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.addf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %199 = tensor.empty() : tensor<1x9x1xf32>
    %200 = linalg.generic {indexing_maps = [#map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198 : tensor<1x9x1xf32>) outs(%199 : tensor<1x9x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %211 = math.rsqrt %in : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x1xf32>
    %201 = tensor.empty() : tensor<1x9x768xf32>
    %202 = linalg.generic {indexing_maps = [#map11, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180, %186 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%201 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.subf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %203 = tensor.empty() : tensor<1x9x768xf32>
    %204 = linalg.generic {indexing_maps = [#map11, #map15, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%202, %200 : tensor<1x9x768xf32>, tensor<1x9x1xf32>) outs(%203 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %expanded_107 = tensor.expand_shape %arg24 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %205 = tensor.empty() : tensor<1x9x768xf32>
    %206 = linalg.generic {indexing_maps = [#map11, #map16, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204, %expanded_107 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%205 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.mulf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %expanded_108 = tensor.expand_shape %arg25 [[0, 1, 2]] : tensor<768xf32> into tensor<1x1x768xf32>
    %207 = tensor.empty() : tensor<1x9x768xf32>
    %208 = linalg.generic {indexing_maps = [#map11, #map16, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%206, %expanded_108 : tensor<1x9x768xf32>, tensor<1x1x768xf32>) outs(%207 : tensor<1x9x768xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %211 = arith.addf %in, %in_109 : f32
      linalg.yield %211 : f32
    } -> tensor<1x9x768xf32>
    %209 = tensor.empty() : tensor<1x9x768xf16>
    %210 = linalg.generic {indexing_maps = [#map11, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%208 : tensor<1x9x768xf32>) outs(%209 : tensor<1x9x768xf16>) {
    ^bb0(%in: f32, %out: f16):
      %211 = arith.truncf %in : f32 to f16
      linalg.yield %211 : f16
    } -> tensor<1x9x768xf16>
    return %210 : tensor<1x9x768xf16>
  }
}

