#!/usr/bin/python
import sys
import os
import re
gGemmParamHeadFile = "CutlassGemmParam.hpp"
gGemmBatchedParamHeadFile = "CutlassGemmBatchedParam.hpp"

def generateGemmFile(headfile):
	hpp = "#ifndef CutlassGemmParam_hpp\n"
	hpp += "#define CutlassGemmParam_hpp\n\n"
	hpp += "#include \"cutlass/epilogue/thread/linear_combination_relu.h\"\n"
	hpp += "#include \"cutlass/epilogue/thread/linear_combination_relu6.h\"\n"
	hpp += "#include \"cutlass/gemm/device/gemm.h\"\n"
	hpp += "#include \"cutlass/gemm/device/gemm_array.h\"\n"
	hpp += "#include \"cutlass/gemm/device/gemm_batched.h\"\n\n"
	hpp += "namespace MNN {\n"
	hpp += "namespace CUDA {\n"
	hpp += "struct CutlassGemmInfo{\n"
	hpp += "    int elh[3];\n"
	hpp += "    int elhPad[3];\n"
	hpp += "};\n\n"

	hpp += "using LayoutInputA = cutlass::layout::RowMajor;\n"
	hpp += "using LayoutInputB = cutlass::layout::ColumnMajor;\n"
	hpp += "using LayoutOutput = cutlass::layout::RowMajor;\n"
	hpp += "using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;\n"
	hpp += "using ElementAccumulator = float;\n"               
	hpp += "using ElementComputeEpilogue = ElementAccumulator;\n"

	hpp += "using ElementInput_F16 = cutlass::half_t;\n"
	hpp += "using ElementInput_F32 = float;\n"
	hpp += "using ElementOutput_F16 = cutlass::half_t;\n"
	hpp += "using ElementOutput_F32 = float;\n\n"

	data_precision = ["cutlass::half_t", "float"]
	processor_type = ["cutlass::arch::OpClassSimt", "cutlass::arch::OpClassTensorOp"]
	sm_arch_type   = ["cutlass::arch::Sm70", "cutlass::arch::Sm75"]
	thread_block_shape = "cutlass::gemm::GemmShape<64, 64, 64>"
	warp_shape     = "cutlass::gemm::GemmShape<32, 32, 64>"
	mma1688_shape  = "cutlass::gemm::GemmShape<16, 8, 8>"
	mma884_shape   = "cutlass::gemm::GemmShape<8, 8, 4>"
	cudaOp_shape   = "cutlass::gemm::GemmShape<1, 1, 1>"
	epilogue_type  = ["LinearCombination", "LinearCombinationRelu", "LinearCombinationRelu6"]

	for epilogue in epilogue_type:
		epilogue_name = ""
		if epilogue == "LinearCombination":
			epilogue_name = "Linear"
		elif epilogue == "LinearCombinationRelu":
			epilogue_name = "Relu"
		elif epilogue == "LinearCombinationRelu6":
			epilogue_name = "Relu6"

		for precision in data_precision:
			precision_name = ""
			if precision == "cutlass::half_t":
				precision_name = "_F16_"
			elif precision == "float":
				precision_name = "_F32_"

			for processor in processor_type:
				processor_name = ""
				if processor == "cutlass::arch::OpClassSimt":
					processor_name = "EpilogueCudaOp"
				elif processor == "cutlass::arch::OpClassTensorOp":
					processor_name = "EpilogueTensorOp"

				hpp += "using " + processor_name + precision_name + epilogue_name  +  " = cutlass::epilogue::thread::" + epilogue + "<\n    "
				hpp += precision + ",\n    "
				if processor == "cutlass::arch::OpClassSimt":
					hpp += "1,\n    "
				elif processor == "cutlass::arch::OpClassTensorOp":
					hpp += "128 / cutlass::sizeof_bits<" + precision + ">::value,\n    "
				hpp += "ElementAccumulator,\n    "
				hpp += "ElementComputeEpilogue>;\n\n"

	hpp += "constexpr int NumStages = 2;\n"

	for epilogue in epilogue_type:
		epilogue_name = ""
		if epilogue == "LinearCombination":
			epilogue_name = "Linear"
		elif epilogue == "LinearCombinationRelu":
			epilogue_name = "Relu"
		elif epilogue == "LinearCombinationRelu6":
			epilogue_name = "Relu6"

		for precision in data_precision:
			inp_precision_name = ""
			if precision == "cutlass::half_t":
				inp_precision_name = "_F16"
			elif precision == "float":
				inp_precision_name = "_F32"

			for processor in processor_type:
				gemm_name = ""
				processor_name = ["EpilogueCudaOp"]
				if processor == "cutlass::arch::OpClassSimt":
					gemm_name = "GemmCuda"
					processor_name = ["EpilogueCudaOp"]
				elif processor == "cutlass::arch::OpClassTensorOp":
					gemm_name = "GemmTensor"
					processor_name = ["EpilogueCudaOp", "EpilogueTensorOp"]


				for smarch in sm_arch_type:
					sm_name = ""
					if gemm_name == "GemmTensor" and smarch == "cutlass::arch::Sm75":
						sm_name = "_Sm75"
					elif gemm_name == "GemmTensor" and smarch == "cutlass::arch::Sm70":
						sm_name = "_Sm70"
					elif gemm_name == "GemmCuda" and smarch == "cutlass::arch::Sm70":
						continue
					element_input_precision = precision
					for element_output_precision in data_precision:
						if element_input_precision == "float" and element_output_precision == "cutlass::half_t":
							continue
						out_precision_name = ""
						if element_output_precision == "cutlass::half_t":
							out_precision_name = "_F16_"
						elif element_output_precision == "float":
							out_precision_name = "_F32_"

						for out_align in processor_name:
							out_align_name = ""
							if out_align == "EpilogueTensorOp":
								out_align_name = "_AlignTensor"
							elif out_align == "EpilogueCudaOp":
								out_align_name = "_AlignCuda"

							hpp += "using " + gemm_name + inp_precision_name + out_precision_name + epilogue_name + out_align_name + sm_name + " = cutlass::gemm::device::Gemm<\n    "
							hpp += element_input_precision + ",\n    LayoutInputA,\n    "
							hpp += element_input_precision + ",\n    LayoutInputB,\n    "
							hpp += element_output_precision + ",\n    LayoutOutput,\n    ElementAccumulator,\n    "
							hpp += processor + ",\n    " + smarch + ",\n    "
							hpp += thread_block_shape + ",\n    " + warp_shape + ",\n    "
							if sm_name == "_Sm75":
								hpp += mma1688_shape + ",\n    "
							elif sm_name == "_Sm70":
								hpp += mma884_shape + ",\n    "
							elif sm_name == "":
								hpp += cudaOp_shape + ",\n    "

							hpp += out_align + out_precision_name + epilogue_name + ",\n    "
							hpp += "SwizzleThreadBlock,\n    "
							hpp += "NumStages"
							if sm_name == "_Sm75":
								hpp += ",\n    128 / cutlass::sizeof_bits<" + element_input_precision + ">::value, 128 / cutlass::sizeof_bits<" + element_input_precision + ">::value, true>;\n\n"
							else :
								hpp += ">;\n\n"

	hpp += "}\n}\n#endif"
	with open(headfile, "w") as f:
		f.write(hpp)

def generateGemmBatchedFile(headfile):
	hpp = "#ifndef CutlassGemmBatchedParam_hpp\n"
	hpp += "#define CutlassGemmBatchedParam_hpp\n\n"
	hpp += "#include \"CutlassGemmParam.hpp\"\n"
	hpp += "#include \"cutlass/gemm/device/gemm_batched.h\"\n\n"
	hpp += "namespace MNN {\n"
	hpp += "namespace CUDA {\n"


	hpp += "using BatchedSwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;\n"
	hpp += "using ShapeBatchMMAThreadBlock = cutlass::gemm::GemmShape<32, 64, 32>;\n"
	hpp += "using ShapeBatchMMAWarp = cutlass::gemm::GemmShape<16, 32, 32>;\n"
	hpp += "using ShapeBatchCudaThreadBlock = cutlass::gemm::GemmShape<64, 64, 64>;\n"
	hpp += "using ShapeBatchCudaWarp = cutlass::gemm::GemmShape<32, 32, 64>;\n\n"

	data_precision = ["cutlass::half_t", "float"]
	processor_type = ["cutlass::arch::OpClassSimt", "cutlass::arch::OpClassTensorOp"]
	sm_arch_type   = ["cutlass::arch::Sm70", "cutlass::arch::Sm75"]
	mma1688_shape  = "cutlass::gemm::GemmShape<16, 8, 8>"
	mma884_shape   = "cutlass::gemm::GemmShape<8, 8, 4>"
	cudaOp_shape   = "cutlass::gemm::GemmShape<1, 1, 1>"
	layout_type    = ["cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor"]
	for element_input_precision in ["cutlass::half_t", "float"]:
		inp_precision_name = ""
		if element_input_precision == "cutlass::half_t":
			inp_precision_name = "_F16"
		elif element_input_precision == "float":
			inp_precision_name = "_F32"

		for processor in processor_type:
			gemm_name = ""
			processor_name = ["EpilogueCudaOp"]
			if processor == "cutlass::arch::OpClassSimt":
				gemm_name = "GemmBatchedCuda"
				processor_name = ["EpilogueCudaOp"]
			elif processor == "cutlass::arch::OpClassTensorOp":
				gemm_name = "GemmBatchedTensor"
				processor_name = ["EpilogueCudaOp", "EpilogueTensorOp"]

			for smarch in sm_arch_type:
				sm_name = ""
				if gemm_name == "GemmBatchedTensor" and smarch == "cutlass::arch::Sm75":
					sm_name = "_Sm75"
				elif gemm_name == "GemmBatchedTensor" and smarch == "cutlass::arch::Sm70":
					sm_name = "_Sm70"
				elif gemm_name == "GemmBatchedCuda" and smarch == "cutlass::arch::Sm70":
					continue

				for element_output_precision in ["cutlass::half_t", "float"]:
					if element_input_precision == "float" and element_output_precision == "cutlass::half_t":
						continue
					out_precision_name = ""
					if element_output_precision == "cutlass::half_t":
						out_precision_name = "_F16_"
					elif element_output_precision == "float":
						out_precision_name = "_F32_"

					for out_align in processor_name:
						out_align_name = ""
						if out_align == "EpilogueTensorOp":
							out_align_name = "_AlignTensor"
						elif out_align == "EpilogueCudaOp":
							out_align_name = "_AlignCuda"

						layout_a = "cutlass::layout::RowMajor"
						layout_a_name = "_Row"
						for layout_b in layout_type:
							layout_b_name = ""
							if layout_b == "cutlass::layout::RowMajor":
								layout_b_name = "_Row"
							elif layout_b == "cutlass::layout::ColumnMajor":
								layout_b_name = "_Column"

							hpp += "using " + gemm_name + inp_precision_name + out_precision_name + "Linear" + out_align_name + layout_a_name + layout_b_name + sm_name + " = cutlass::gemm::device::GemmBatched<\n    "
							hpp += element_input_precision + ",\n    " + layout_a + ",\n    "
							hpp += element_input_precision + ",\n    " + layout_b + ",\n    "
							hpp += element_output_precision + ",\n    LayoutOutput,\n    ElementAccumulator,\n    "
							hpp += processor + ",\n    " + smarch + ",\n    "
							if gemm_name == "GemmBatchedTensor":
								hpp += "ShapeBatchMMAThreadBlock,\n    ShapeBatchMMAWarp,\n    "
							elif gemm_name == "GemmBatchedCuda":
								hpp += "ShapeBatchCudaThreadBlock,\n    ShapeBatchCudaWarp,\n    "
							if sm_name == "_Sm75":
								hpp += mma1688_shape + ",\n    "
							elif sm_name == "_Sm70":
								hpp += mma884_shape + ",\n    "
							elif sm_name == "":
								hpp += cudaOp_shape + ",\n    "

							hpp += out_align + out_precision_name + "Linear,\n    "
							hpp += "BatchedSwizzleThreadBlock,\n    "
							hpp += "NumStages>;\n\n"

	hpp += "}\n}\n#endif"
	with open(headfile, "w") as f:
		f.write(hpp)

if __name__ == '__main__':
    generateGemmFile(gGemmParamHeadFile);
    generateGemmBatchedFile(gGemmBatchedParamHeadFile);

