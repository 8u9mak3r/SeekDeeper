#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @subgraph0(%arg0: tensor<1x9x768xf16>, %arg1: tensor<1x9x768xf16>, %arg2: tensor<768x768xi8>, %arg3: tensor<768xf16>, %arg4: tensor<768xf16>, %arg5: tensor<1x9x768xf16>, %arg6: tensor<1x9x768xf16>, %arg7: tensor<768x768xi8>, %arg8: tensor<768xf16>, %arg9: tensor<768xf16>, %arg10: tensor<1x9x768xf16>, %arg11: tensor<1x9x768xf16>, %arg12: tensor<768x768xi8>, %arg13: tensor<768xf16>, %arg14: tensor<768xf16>, %arg15: tensor<768x768xi8>, %arg16: tensor<768xf16>, %arg17: tensor<768xf16>, %arg18: tensor<1x9x768xf16>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<3072x768xi8>, %arg22: tensor<3072xf16>, %arg23: tensor<3072xf16>, %arg24: tensor<768x3072xi8>, %arg25: tensor<768xf16>, %arg26: tensor<768xf16>, %arg27: tensor<768xf32>, %arg28: tensor<768xf32>) -> tensor<1x9x768xf16> {
    %0 = tosa.reshape %arg18 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %1 = tensor.empty() : tensor<1x9xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<1x9x768xf16>) outs(%1 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.minimumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %3 = tensor.empty() : tensor<1x9xf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<1x9x768xf16>) outs(%3 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.maximumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %5 = tosa.minimum %2, %cst : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %6 = tosa.maximum %4, %cst_0 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %7 = tensor.empty() : tensor<1x9xf16>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x9xf16>) outs(%7 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.negf %in : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %9 = tosa.maximum %8, %6 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_1 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %10 = tosa.reshape %cst_1 {new_shape = array<i64: 1, 1>} : (tensor<1xf16>) -> tensor<1x1xf16>
    %11 = tosa.reciprocal %10 : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %12 = tosa.mul %9, %11 {shift = 0 : i8} : (tensor<1x9xf16>, tensor<1x1xf16>) -> tensor<1x9xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %13 = tosa.cast %12 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %14 = tosa.clamp %13 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 9.99999974E-6 : f32, min_int = 0 : i64} : (tensor<1x9xf32>) -> tensor<1x9xf32>
    %15 = tosa.cast %14 : (tensor<1x9xf32>) -> tensor<1x9xf16>
    %16 = tosa.cast %15 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %17 = tosa.reshape %arg18 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %18 = tosa.reshape %16 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %19 = tosa.reshape %cst_2 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf16>) -> tensor<1x9x1xf16>
    %20 = tosa.reciprocal %18 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %21 = tosa.reshape %cst_3 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %22 = tosa.mul %20, %21 {shift = 0 : i8} : (tensor<1x9x1xf32>, tensor<1x1x1xf32>) -> tensor<1x9x1xf32>
    %23 = tosa.cast %17 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %24 = tosa.mul %23, %22 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %25 = tosa.floor %24 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %26 = tosa.cast %19 : (tensor<1x9x1xf16>) -> tensor<1x9x1xf32>
    %27 = tosa.add %25, %26 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %28 = tosa.clamp %27 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = -1.270000e+02 : f32, min_int = -127 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %29 = tosa.clamp %28 {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %30 = tosa.reshape %29 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %31 = tensor.empty() : tensor<1x9x768xi8>
    %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<1x9x768xf32>) outs(%31 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %381 = arith.fptosi %in : f32 to i8
      linalg.yield %381 : i8
    } -> tensor<1x9x768xi8>
    %33 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %34 = tosa.transpose %arg2, %33 : (tensor<768x768xi8>, tensor<2xi32>) -> tensor<768x768xi8>
    %35 = tosa.reshape %32 {new_shape = array<i64: 9, 768>} : (tensor<1x9x768xi8>) -> tensor<9x768xi8>
    %36 = tosa.reshape %16 {new_shape = array<i64: 9, 1>} : (tensor<1x9xf32>) -> tensor<9x1xf32>
    %37 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<9x768xf32>}> : () -> tensor<9x768xf32>
    %38 = tosa.add %36, %37 : (tensor<9x1xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %cst_4 = arith.constant dense<0> : tensor<9x768xi32>
    %39 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%35, %34 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_4 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %40 = tosa.cast %39 : (tensor<9x768xi32>) -> tensor<9x768xf32>
    %41 = tosa.mul %40, %38 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %42 = tosa.cast %arg3 : (tensor<768xf16>) -> tensor<768xf32>
    %43 = tosa.reshape %42 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %44 = tosa.mul %41, %43 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<1x768xf32>) -> tensor<9x768xf32>
    %45 = tosa.reshape %44 {new_shape = array<i64: 1, 9, 768>} : (tensor<9x768xf32>) -> tensor<1x9x768xf32>
    %46 = tosa.cast %45 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    %47 = tosa.reshape %arg4 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %48 = tosa.add %46, %47 : (tensor<1x9x768xf16>, tensor<1x1x768xf16>) -> tensor<1x9x768xf16>
    %49 = tosa.reshape %arg18 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %50 = tensor.empty() : tensor<1x9xf16>
    %51 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : tensor<1x9x768xf16>) outs(%50 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.minimumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %52 = tensor.empty() : tensor<1x9xf16>
    %53 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : tensor<1x9x768xf16>) outs(%52 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.maximumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %54 = tosa.minimum %51, %cst_5 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %55 = tosa.maximum %53, %cst_6 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %56 = tensor.empty() : tensor<1x9xf16>
    %57 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%54 : tensor<1x9xf16>) outs(%56 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.negf %in : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %58 = tosa.maximum %57, %55 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_7 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %59 = tosa.reshape %cst_7 {new_shape = array<i64: 1, 1>} : (tensor<1xf16>) -> tensor<1x1xf16>
    %60 = tosa.reciprocal %59 : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %61 = tosa.mul %58, %60 {shift = 0 : i8} : (tensor<1x9xf16>, tensor<1x1xf16>) -> tensor<1x9xf16>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %62 = tosa.cast %61 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %63 = tosa.clamp %62 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 9.99999974E-6 : f32, min_int = 0 : i64} : (tensor<1x9xf32>) -> tensor<1x9xf32>
    %64 = tosa.cast %63 : (tensor<1x9xf32>) -> tensor<1x9xf16>
    %65 = tosa.cast %64 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %66 = tosa.reshape %arg18 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %67 = tosa.reshape %65 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %68 = tosa.reshape %cst_8 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf16>) -> tensor<1x9x1xf16>
    %69 = tosa.reciprocal %67 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %70 = tosa.reshape %cst_9 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %71 = tosa.mul %69, %70 {shift = 0 : i8} : (tensor<1x9x1xf32>, tensor<1x1x1xf32>) -> tensor<1x9x1xf32>
    %72 = tosa.cast %66 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %73 = tosa.mul %72, %71 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %74 = tosa.floor %73 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %75 = tosa.cast %68 : (tensor<1x9x1xf16>) -> tensor<1x9x1xf32>
    %76 = tosa.add %74, %75 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %77 = tosa.clamp %76 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = -1.270000e+02 : f32, min_int = -127 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %78 = tosa.clamp %77 {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %79 = tosa.reshape %78 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %80 = tensor.empty() : tensor<1x9x768xi8>
    %81 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79 : tensor<1x9x768xf32>) outs(%80 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %381 = arith.fptosi %in : f32 to i8
      linalg.yield %381 : i8
    } -> tensor<1x9x768xi8>
    %82 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %83 = tosa.transpose %arg7, %82 : (tensor<768x768xi8>, tensor<2xi32>) -> tensor<768x768xi8>
    %84 = tosa.reshape %81 {new_shape = array<i64: 9, 768>} : (tensor<1x9x768xi8>) -> tensor<9x768xi8>
    %85 = tosa.reshape %65 {new_shape = array<i64: 9, 1>} : (tensor<1x9xf32>) -> tensor<9x1xf32>
    %86 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<9x768xf32>}> : () -> tensor<9x768xf32>
    %87 = tosa.add %85, %86 : (tensor<9x1xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %cst_10 = arith.constant dense<0> : tensor<9x768xi32>
    %88 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%84, %83 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_10 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %89 = tosa.cast %88 : (tensor<9x768xi32>) -> tensor<9x768xf32>
    %90 = tosa.mul %89, %87 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %91 = tosa.cast %arg8 : (tensor<768xf16>) -> tensor<768xf32>
    %92 = tosa.reshape %91 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %93 = tosa.mul %90, %92 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<1x768xf32>) -> tensor<9x768xf32>
    %94 = tosa.reshape %93 {new_shape = array<i64: 1, 9, 768>} : (tensor<9x768xf32>) -> tensor<1x9x768xf32>
    %95 = tosa.cast %94 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    %96 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %97 = tosa.add %95, %96 : (tensor<1x9x768xf16>, tensor<1x1x768xf16>) -> tensor<1x9x768xf16>
    %98 = tosa.reshape %arg18 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %99 = tensor.empty() : tensor<1x9xf16>
    %100 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%98 : tensor<1x9x768xf16>) outs(%99 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.minimumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %101 = tensor.empty() : tensor<1x9xf16>
    %102 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%98 : tensor<1x9x768xf16>) outs(%101 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.maximumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %103 = tosa.minimum %100, %cst_11 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %104 = tosa.maximum %102, %cst_12 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %105 = tensor.empty() : tensor<1x9xf16>
    %106 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%103 : tensor<1x9xf16>) outs(%105 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.negf %in : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %107 = tosa.maximum %106, %104 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_13 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %108 = tosa.reshape %cst_13 {new_shape = array<i64: 1, 1>} : (tensor<1xf16>) -> tensor<1x1xf16>
    %109 = tosa.reciprocal %108 : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %110 = tosa.mul %107, %109 {shift = 0 : i8} : (tensor<1x9xf16>, tensor<1x1xf16>) -> tensor<1x9xf16>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %111 = tosa.cast %110 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %112 = tosa.clamp %111 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 9.99999974E-6 : f32, min_int = 0 : i64} : (tensor<1x9xf32>) -> tensor<1x9xf32>
    %113 = tosa.cast %112 : (tensor<1x9xf32>) -> tensor<1x9xf16>
    %114 = tosa.cast %113 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %115 = tosa.reshape %arg18 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %116 = tosa.reshape %114 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %117 = tosa.reshape %cst_14 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf16>) -> tensor<1x9x1xf16>
    %118 = tosa.reciprocal %116 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %cst_15 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %119 = tosa.reshape %cst_15 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %120 = tosa.mul %118, %119 {shift = 0 : i8} : (tensor<1x9x1xf32>, tensor<1x1x1xf32>) -> tensor<1x9x1xf32>
    %121 = tosa.cast %115 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %122 = tosa.mul %121, %120 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %123 = tosa.floor %122 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %124 = tosa.cast %117 : (tensor<1x9x1xf16>) -> tensor<1x9x1xf32>
    %125 = tosa.add %123, %124 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %126 = tosa.clamp %125 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = -1.270000e+02 : f32, min_int = -127 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %127 = tosa.clamp %126 {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %128 = tosa.reshape %127 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %129 = tensor.empty() : tensor<1x9x768xi8>
    %130 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128 : tensor<1x9x768xf32>) outs(%129 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %381 = arith.fptosi %in : f32 to i8
      linalg.yield %381 : i8
    } -> tensor<1x9x768xi8>
    %131 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %132 = tosa.transpose %arg12, %131 : (tensor<768x768xi8>, tensor<2xi32>) -> tensor<768x768xi8>
    %133 = tosa.reshape %130 {new_shape = array<i64: 9, 768>} : (tensor<1x9x768xi8>) -> tensor<9x768xi8>
    %134 = tosa.reshape %114 {new_shape = array<i64: 9, 1>} : (tensor<1x9xf32>) -> tensor<9x1xf32>
    %135 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<9x768xf32>}> : () -> tensor<9x768xf32>
    %136 = tosa.add %134, %135 : (tensor<9x1xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %cst_16 = arith.constant dense<0> : tensor<9x768xi32>
    %137 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%133, %132 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_16 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %138 = tosa.cast %137 : (tensor<9x768xi32>) -> tensor<9x768xf32>
    %139 = tosa.mul %138, %136 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %140 = tosa.cast %arg13 : (tensor<768xf16>) -> tensor<768xf32>
    %141 = tosa.reshape %140 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %142 = tosa.mul %139, %141 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<1x768xf32>) -> tensor<9x768xf32>
    %143 = tosa.reshape %142 {new_shape = array<i64: 1, 9, 768>} : (tensor<9x768xf32>) -> tensor<1x9x768xf32>
    %144 = tosa.cast %143 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    %145 = tosa.reshape %arg14 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %146 = tosa.add %144, %145 : (tensor<1x9x768xf16>, tensor<1x1x768xf16>) -> tensor<1x9x768xf16>
    %147 = tosa.reshape %48 {new_shape = array<i64: 1, 9, 12, 64>} : (tensor<1x9x768xf16>) -> tensor<1x9x12x64xf16>
    %148 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %149 = tosa.transpose %147, %148 : (tensor<1x9x12x64xf16>, tensor<4xi32>) -> tensor<1x12x9x64xf16>
    %150 = tosa.reshape %97 {new_shape = array<i64: 1, 9, 12, 64>} : (tensor<1x9x768xf16>) -> tensor<1x9x12x64xf16>
    %151 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %152 = tosa.transpose %150, %151 : (tensor<1x9x12x64xf16>, tensor<4xi32>) -> tensor<1x12x9x64xf16>
    %153 = tosa.reshape %146 {new_shape = array<i64: 1, 9, 12, 64>} : (tensor<1x9x768xf16>) -> tensor<1x9x12x64xf16>
    %154 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %155 = tosa.transpose %153, %154 : (tensor<1x9x12x64xf16>, tensor<4xi32>) -> tensor<1x12x9x64xf16>
    %cst_17 = arith.constant 0.000000e+00 : f16
    %splat = tensor.splat %cst_17 : tensor<9x9xf16>
    %156 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %157 = tosa.transpose %152, %156 : (tensor<1x12x9x64xf16>, tensor<4xi32>) -> tensor<1x12x64x9xf16>
    %158 = tosa.reshape %149 {new_shape = array<i64: 12, 9, 64>} : (tensor<1x12x9x64xf16>) -> tensor<12x9x64xf16>
    %159 = tosa.reshape %157 {new_shape = array<i64: 12, 64, 9>} : (tensor<1x12x64x9xf16>) -> tensor<12x64x9xf16>
    %160 = tosa.matmul %158, %159 : (tensor<12x9x64xf16>, tensor<12x64x9xf16>) -> tensor<12x9x9xf16>
    %cst_18 = arith.constant 1.250000e-01 : f16
    %splat_19 = tensor.splat %cst_18 : tensor<12x9x9xf16>
    %161 = tosa.mul %160, %splat_19 {shift = 0 : i8} : (tensor<12x9x9xf16>, tensor<12x9x9xf16>) -> tensor<12x9x9xf16>
    %162 = tosa.add %161, %splat : (tensor<12x9x9xf16>, tensor<9x9xf16>) -> tensor<12x9x9xf16>
    %163 = tosa.reduce_max %162 {axis = 2 : i32} : (tensor<12x9x9xf16>) -> tensor<12x9x1xf16>
    %164 = tosa.sub %162, %163 : (tensor<12x9x9xf16>, tensor<12x9x1xf16>) -> tensor<12x9x9xf16>
    %165 = math.exp %164 : tensor<12x9x9xf16>
    %166 = tosa.reduce_sum %165 {axis = 2 : i32} : (tensor<12x9x9xf16>) -> tensor<12x9x1xf16>
    %167 = tosa.log %166 : (tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    %168 = tosa.add %163, %167 : (tensor<12x9x1xf16>, tensor<12x9x1xf16>) -> tensor<12x9x1xf16>
    %169 = tosa.sub %162, %168 : (tensor<12x9x9xf16>, tensor<12x9x1xf16>) -> tensor<12x9x9xf16>
    %170 = math.exp %169 : tensor<12x9x9xf16>
    %171 = tosa.reshape %168 {new_shape = array<i64: 1, 12, 9>} : (tensor<12x9x1xf16>) -> tensor<1x12x9xf16>
    %172 = tosa.reshape %155 {new_shape = array<i64: 12, 9, 64>} : (tensor<1x12x9x64xf16>) -> tensor<12x9x64xf16>
    %173 = tosa.matmul %170, %172 : (tensor<12x9x9xf16>, tensor<12x9x64xf16>) -> tensor<12x9x64xf16>
    %174 = tosa.reshape %173 {new_shape = array<i64: 1, 12, 9, 64>} : (tensor<12x9x64xf16>) -> tensor<1x12x9x64xf16>
    %175 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %176 = tosa.transpose %174, %175 : (tensor<1x12x9x64xf16>, tensor<4xi32>) -> tensor<1x9x12x64xf16>
    %177 = tosa.reshape %176 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x12x64xf16>) -> tensor<1x9x768xf16>
    %178 = tosa.reshape %177 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %179 = tensor.empty() : tensor<1x9xf16>
    %180 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%178 : tensor<1x9x768xf16>) outs(%179 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.minimumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %181 = tensor.empty() : tensor<1x9xf16>
    %182 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%178 : tensor<1x9x768xf16>) outs(%181 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.maximumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %cst_20 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %183 = tosa.minimum %180, %cst_20 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_21 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %184 = tosa.maximum %182, %cst_21 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %185 = tensor.empty() : tensor<1x9xf16>
    %186 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%183 : tensor<1x9xf16>) outs(%185 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.negf %in : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %187 = tosa.maximum %186, %184 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_22 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %188 = tosa.reshape %cst_22 {new_shape = array<i64: 1, 1>} : (tensor<1xf16>) -> tensor<1x1xf16>
    %189 = tosa.reciprocal %188 : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %190 = tosa.mul %187, %189 {shift = 0 : i8} : (tensor<1x9xf16>, tensor<1x1xf16>) -> tensor<1x9xf16>
    %cst_23 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %191 = tosa.cast %190 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %192 = tosa.clamp %191 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 9.99999974E-6 : f32, min_int = 0 : i64} : (tensor<1x9xf32>) -> tensor<1x9xf32>
    %193 = tosa.cast %192 : (tensor<1x9xf32>) -> tensor<1x9xf16>
    %194 = tosa.cast %193 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %195 = tosa.reshape %177 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %196 = tosa.reshape %194 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %197 = tosa.reshape %cst_23 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf16>) -> tensor<1x9x1xf16>
    %198 = tosa.reciprocal %196 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %cst_24 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %199 = tosa.reshape %cst_24 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %200 = tosa.mul %198, %199 {shift = 0 : i8} : (tensor<1x9x1xf32>, tensor<1x1x1xf32>) -> tensor<1x9x1xf32>
    %201 = tosa.cast %195 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %202 = tosa.mul %201, %200 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %203 = tosa.floor %202 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %204 = tosa.cast %197 : (tensor<1x9x1xf16>) -> tensor<1x9x1xf32>
    %205 = tosa.add %203, %204 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %206 = tosa.clamp %205 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = -1.270000e+02 : f32, min_int = -127 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %207 = tosa.clamp %206 {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %208 = tosa.reshape %207 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %209 = tensor.empty() : tensor<1x9x768xi8>
    %210 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%208 : tensor<1x9x768xf32>) outs(%209 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %381 = arith.fptosi %in : f32 to i8
      linalg.yield %381 : i8
    } -> tensor<1x9x768xi8>
    %211 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %212 = tosa.transpose %arg15, %211 : (tensor<768x768xi8>, tensor<2xi32>) -> tensor<768x768xi8>
    %213 = tosa.reshape %210 {new_shape = array<i64: 9, 768>} : (tensor<1x9x768xi8>) -> tensor<9x768xi8>
    %214 = tosa.reshape %194 {new_shape = array<i64: 9, 1>} : (tensor<1x9xf32>) -> tensor<9x1xf32>
    %215 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<9x768xf32>}> : () -> tensor<9x768xf32>
    %216 = tosa.add %214, %215 : (tensor<9x1xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %cst_25 = arith.constant dense<0> : tensor<9x768xi32>
    %217 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%213, %212 : tensor<9x768xi8>, tensor<768x768xi8>) outs(%cst_25 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %218 = tosa.cast %217 : (tensor<9x768xi32>) -> tensor<9x768xf32>
    %219 = tosa.mul %218, %216 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %220 = tosa.cast %arg16 : (tensor<768xf16>) -> tensor<768xf32>
    %221 = tosa.reshape %220 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %222 = tosa.mul %219, %221 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<1x768xf32>) -> tensor<9x768xf32>
    %223 = tosa.reshape %222 {new_shape = array<i64: 1, 9, 768>} : (tensor<9x768xf32>) -> tensor<1x9x768xf32>
    %224 = tosa.cast %223 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    %225 = tosa.reshape %arg17 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %226 = tosa.add %224, %225 : (tensor<1x9x768xf16>, tensor<1x1x768xf16>) -> tensor<1x9x768xf16>
    %227 = tosa.identity %226 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %228 = tosa.add %arg18, %227 : (tensor<1x9x768xf16>, tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %229 = tosa.cast %228 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %230 = tosa.reduce_sum %229 {axis = 2 : i32} : (tensor<1x9x768xf32>) -> tensor<1x9x1xf32>
    %231 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %232 = tosa.reciprocal %231 : (tensor<1xf32>) -> tensor<1xf32>
    %233 = tosa.mul %232, %230 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %234 = tosa.sub %229, %233 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %235 = tosa.mul %234, %234 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %236 = tosa.reduce_sum %235 {axis = 2 : i32} : (tensor<1x9x768xf32>) -> tensor<1x9x1xf32>
    %237 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %238 = tosa.reciprocal %237 : (tensor<1xf32>) -> tensor<1xf32>
    %239 = tosa.mul %238, %236 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %240 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x9x1xf32>}> : () -> tensor<1x9x1xf32>
    %241 = tosa.add %239, %240 : (tensor<1x9x1xf32>, tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %242 = tosa.rsqrt %241 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %243 = tosa.sub %229, %233 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %244 = tosa.mul %243, %242 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %245 = tosa.reshape %arg19 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %246 = tosa.mul %244, %245 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x1x768xf32>) -> tensor<1x9x768xf32>
    %247 = tosa.reshape %arg20 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %248 = tosa.add %246, %247 : (tensor<1x9x768xf32>, tensor<1x1x768xf32>) -> tensor<1x9x768xf32>
    %249 = tosa.cast %248 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    %250 = tosa.reshape %249 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %251 = tensor.empty() : tensor<1x9xf16>
    %252 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%250 : tensor<1x9x768xf16>) outs(%251 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.minimumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %253 = tensor.empty() : tensor<1x9xf16>
    %254 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%250 : tensor<1x9x768xf16>) outs(%253 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.maximumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %255 = tosa.minimum %252, %cst_26 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_27 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %256 = tosa.maximum %254, %cst_27 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %257 = tensor.empty() : tensor<1x9xf16>
    %258 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%255 : tensor<1x9xf16>) outs(%257 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.negf %in : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %259 = tosa.maximum %258, %256 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_28 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %260 = tosa.reshape %cst_28 {new_shape = array<i64: 1, 1>} : (tensor<1xf16>) -> tensor<1x1xf16>
    %261 = tosa.reciprocal %260 : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %262 = tosa.mul %259, %261 {shift = 0 : i8} : (tensor<1x9xf16>, tensor<1x1xf16>) -> tensor<1x9xf16>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %263 = tosa.cast %262 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %264 = tosa.clamp %263 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 9.99999974E-6 : f32, min_int = 0 : i64} : (tensor<1x9xf32>) -> tensor<1x9xf32>
    %265 = tosa.cast %264 : (tensor<1x9xf32>) -> tensor<1x9xf16>
    %266 = tosa.cast %265 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %267 = tosa.reshape %249 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %268 = tosa.reshape %266 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %269 = tosa.reshape %cst_29 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf16>) -> tensor<1x9x1xf16>
    %270 = tosa.reciprocal %268 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %cst_30 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %271 = tosa.reshape %cst_30 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %272 = tosa.mul %270, %271 {shift = 0 : i8} : (tensor<1x9x1xf32>, tensor<1x1x1xf32>) -> tensor<1x9x1xf32>
    %273 = tosa.cast %267 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %274 = tosa.mul %273, %272 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %275 = tosa.floor %274 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %276 = tosa.cast %269 : (tensor<1x9x1xf16>) -> tensor<1x9x1xf32>
    %277 = tosa.add %275, %276 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %278 = tosa.clamp %277 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = -1.270000e+02 : f32, min_int = -127 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %279 = tosa.clamp %278 {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %280 = tosa.reshape %279 {new_shape = array<i64: 1, 9, 768>} : (tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %281 = tensor.empty() : tensor<1x9x768xi8>
    %282 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%280 : tensor<1x9x768xf32>) outs(%281 : tensor<1x9x768xi8>) {
    ^bb0(%in: f32, %out: i8):
      %381 = arith.fptosi %in : f32 to i8
      linalg.yield %381 : i8
    } -> tensor<1x9x768xi8>
    %283 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %284 = tosa.transpose %arg21, %283 : (tensor<3072x768xi8>, tensor<2xi32>) -> tensor<768x3072xi8>
    %285 = tosa.reshape %282 {new_shape = array<i64: 9, 768>} : (tensor<1x9x768xi8>) -> tensor<9x768xi8>
    %286 = tosa.reshape %266 {new_shape = array<i64: 9, 1>} : (tensor<1x9xf32>) -> tensor<9x1xf32>
    %287 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<9x3072xf32>}> : () -> tensor<9x3072xf32>
    %288 = tosa.add %286, %287 : (tensor<9x1xf32>, tensor<9x3072xf32>) -> tensor<9x3072xf32>
    %cst_31 = arith.constant dense<0> : tensor<9x3072xi32>
    %289 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%285, %284 : tensor<9x768xi8>, tensor<768x3072xi8>) outs(%cst_31 : tensor<9x3072xi32>) -> tensor<9x3072xi32>
    %290 = tosa.cast %289 : (tensor<9x3072xi32>) -> tensor<9x3072xf32>
    %291 = tosa.mul %290, %288 {shift = 0 : i8} : (tensor<9x3072xf32>, tensor<9x3072xf32>) -> tensor<9x3072xf32>
    %292 = tosa.cast %arg22 : (tensor<3072xf16>) -> tensor<3072xf32>
    %293 = tosa.reshape %292 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %294 = tosa.mul %291, %293 {shift = 0 : i8} : (tensor<9x3072xf32>, tensor<1x3072xf32>) -> tensor<9x3072xf32>
    %295 = tosa.reshape %294 {new_shape = array<i64: 1, 9, 3072>} : (tensor<9x3072xf32>) -> tensor<1x9x3072xf32>
    %296 = tosa.cast %295 : (tensor<1x9x3072xf32>) -> tensor<1x9x3072xf16>
    %297 = tosa.reshape %arg23 {new_shape = array<i64: 1, 1, 3072>} : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %298 = tosa.add %296, %297 : (tensor<1x9x3072xf16>, tensor<1x1x3072xf16>) -> tensor<1x9x3072xf16>
    %299 = tosa.cast %298 : (tensor<1x9x3072xf16>) -> tensor<1x9x3072xf32>
    %cst_32 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %300 = tosa.reshape %cst_32 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %301 = tosa.mul %299, %300 {shift = 0 : i8} : (tensor<1x9x3072xf32>, tensor<1x1x1xf32>) -> tensor<1x9x3072xf32>
    %cst_33 = arith.constant dense<0.707106769> : tensor<1xf32>
    %302 = tosa.reshape %cst_33 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %303 = tosa.mul %299, %302 {shift = 0 : i8} : (tensor<1x9x3072xf32>, tensor<1x1x1xf32>) -> tensor<1x9x3072xf32>
    %304 = math.erf %303 : tensor<1x9x3072xf32>
    %305 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x9x3072xf32>}> : () -> tensor<1x9x3072xf32>
    %306 = tosa.add %304, %305 : (tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) -> tensor<1x9x3072xf32>
    %307 = tosa.mul %301, %306 {shift = 0 : i8} : (tensor<1x9x3072xf32>, tensor<1x9x3072xf32>) -> tensor<1x9x3072xf32>
    %308 = tosa.cast %307 : (tensor<1x9x3072xf32>) -> tensor<1x9x3072xf16>
    %309 = tosa.reshape %308 {new_shape = array<i64: 1, 9, 3072>} : (tensor<1x9x3072xf16>) -> tensor<1x9x3072xf16>
    %310 = tensor.empty() : tensor<1x9xf16>
    %311 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%309 : tensor<1x9x3072xf16>) outs(%310 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.minimumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %312 = tensor.empty() : tensor<1x9xf16>
    %313 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%309 : tensor<1x9x3072xf16>) outs(%312 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.maximumf %in, %out : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %cst_34 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %314 = tosa.minimum %311, %cst_34 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_35 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %315 = tosa.maximum %313, %cst_35 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %316 = tensor.empty() : tensor<1x9xf16>
    %317 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%314 : tensor<1x9xf16>) outs(%316 : tensor<1x9xf16>) {
    ^bb0(%in: f16, %out: f16):
      %381 = arith.negf %in : f16
      linalg.yield %381 : f16
    } -> tensor<1x9xf16>
    %318 = tosa.maximum %317, %315 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %cst_36 = arith.constant dense<1.270000e+02> : tensor<1xf16>
    %319 = tosa.reshape %cst_36 {new_shape = array<i64: 1, 1>} : (tensor<1xf16>) -> tensor<1x1xf16>
    %320 = tosa.reciprocal %319 : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %321 = tosa.mul %318, %320 {shift = 0 : i8} : (tensor<1x9xf16>, tensor<1x1xf16>) -> tensor<1x9xf16>
    %cst_37 = arith.constant dense<0.000000e+00> : tensor<1x9xf16>
    %322 = tosa.cast %321 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %323 = tosa.clamp %322 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 9.99999974E-6 : f32, min_int = 0 : i64} : (tensor<1x9xf32>) -> tensor<1x9xf32>
    %324 = tosa.cast %323 : (tensor<1x9xf32>) -> tensor<1x9xf16>
    %325 = tosa.cast %324 : (tensor<1x9xf16>) -> tensor<1x9xf32>
    %326 = tosa.reshape %308 {new_shape = array<i64: 1, 9, 3072>} : (tensor<1x9x3072xf16>) -> tensor<1x9x3072xf16>
    %327 = tosa.reshape %325 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %328 = tosa.reshape %cst_37 {new_shape = array<i64: 1, 9, 1>} : (tensor<1x9xf16>) -> tensor<1x9x1xf16>
    %329 = tosa.reciprocal %327 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %cst_38 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %330 = tosa.reshape %cst_38 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %331 = tosa.mul %329, %330 {shift = 0 : i8} : (tensor<1x9x1xf32>, tensor<1x1x1xf32>) -> tensor<1x9x1xf32>
    %332 = tosa.cast %326 : (tensor<1x9x3072xf16>) -> tensor<1x9x3072xf32>
    %333 = tosa.mul %332, %331 {shift = 0 : i8} : (tensor<1x9x3072xf32>, tensor<1x9x1xf32>) -> tensor<1x9x3072xf32>
    %334 = tosa.floor %333 : (tensor<1x9x3072xf32>) -> tensor<1x9x3072xf32>
    %335 = tosa.cast %328 : (tensor<1x9x1xf16>) -> tensor<1x9x1xf32>
    %336 = tosa.add %334, %335 : (tensor<1x9x3072xf32>, tensor<1x9x1xf32>) -> tensor<1x9x3072xf32>
    %337 = tosa.clamp %336 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = -1.270000e+02 : f32, min_int = -127 : i64} : (tensor<1x9x3072xf32>) -> tensor<1x9x3072xf32>
    %338 = tosa.clamp %337 {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x9x3072xf32>) -> tensor<1x9x3072xf32>
    %339 = tosa.reshape %338 {new_shape = array<i64: 1, 9, 3072>} : (tensor<1x9x3072xf32>) -> tensor<1x9x3072xf32>
    %340 = tensor.empty() : tensor<1x9x3072xi8>
    %341 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%339 : tensor<1x9x3072xf32>) outs(%340 : tensor<1x9x3072xi8>) {
    ^bb0(%in: f32, %out: i8):
      %381 = arith.fptosi %in : f32 to i8
      linalg.yield %381 : i8
    } -> tensor<1x9x3072xi8>
    %342 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %343 = tosa.transpose %arg24, %342 : (tensor<768x3072xi8>, tensor<2xi32>) -> tensor<3072x768xi8>
    %344 = tosa.reshape %341 {new_shape = array<i64: 9, 3072>} : (tensor<1x9x3072xi8>) -> tensor<9x3072xi8>
    %345 = tosa.reshape %325 {new_shape = array<i64: 9, 1>} : (tensor<1x9xf32>) -> tensor<9x1xf32>
    %346 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<9x768xf32>}> : () -> tensor<9x768xf32>
    %347 = tosa.add %345, %346 : (tensor<9x1xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %cst_39 = arith.constant dense<0> : tensor<9x768xi32>
    %348 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%344, %343 : tensor<9x3072xi8>, tensor<3072x768xi8>) outs(%cst_39 : tensor<9x768xi32>) -> tensor<9x768xi32>
    %349 = tosa.cast %348 : (tensor<9x768xi32>) -> tensor<9x768xf32>
    %350 = tosa.mul %349, %347 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<9x768xf32>) -> tensor<9x768xf32>
    %351 = tosa.cast %arg25 : (tensor<768xf16>) -> tensor<768xf32>
    %352 = tosa.reshape %351 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %353 = tosa.mul %350, %352 {shift = 0 : i8} : (tensor<9x768xf32>, tensor<1x768xf32>) -> tensor<9x768xf32>
    %354 = tosa.reshape %353 {new_shape = array<i64: 1, 9, 768>} : (tensor<9x768xf32>) -> tensor<1x9x768xf32>
    %355 = tosa.cast %354 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    %356 = tosa.reshape %arg26 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %357 = tosa.add %355, %356 : (tensor<1x9x768xf16>, tensor<1x1x768xf16>) -> tensor<1x9x768xf16>
    %358 = tosa.identity %357 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %359 = tosa.add %249, %358 : (tensor<1x9x768xf16>, tensor<1x9x768xf16>) -> tensor<1x9x768xf16>
    %360 = tosa.cast %359 : (tensor<1x9x768xf16>) -> tensor<1x9x768xf32>
    %361 = tosa.reduce_sum %360 {axis = 2 : i32} : (tensor<1x9x768xf32>) -> tensor<1x9x1xf32>
    %362 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %363 = tosa.reciprocal %362 : (tensor<1xf32>) -> tensor<1xf32>
    %364 = tosa.mul %363, %361 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %365 = tosa.sub %360, %364 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %366 = tosa.mul %365, %365 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x768xf32>) -> tensor<1x9x768xf32>
    %367 = tosa.reduce_sum %366 {axis = 2 : i32} : (tensor<1x9x768xf32>) -> tensor<1x9x1xf32>
    %368 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %369 = tosa.reciprocal %368 : (tensor<1xf32>) -> tensor<1xf32>
    %370 = tosa.mul %369, %367 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %371 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x9x1xf32>}> : () -> tensor<1x9x1xf32>
    %372 = tosa.add %370, %371 : (tensor<1x9x1xf32>, tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %373 = tosa.rsqrt %372 : (tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %374 = tosa.sub %360, %364 : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %375 = tosa.mul %374, %373 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x9x1xf32>) -> tensor<1x9x768xf32>
    %376 = tosa.reshape %arg27 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %377 = tosa.mul %375, %376 {shift = 0 : i8} : (tensor<1x9x768xf32>, tensor<1x1x768xf32>) -> tensor<1x9x768xf32>
    %378 = tosa.reshape %arg28 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %379 = tosa.add %377, %378 : (tensor<1x9x768xf32>, tensor<1x1x768xf32>) -> tensor<1x9x768xf32>
    %380 = tosa.cast %379 : (tensor<1x9x768xf32>) -> tensor<1x9x768xf16>
    return %380 : tensor<1x9x768xf16>
  }
}

