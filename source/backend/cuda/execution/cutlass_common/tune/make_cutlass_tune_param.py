#!/usr/bin/python
import sys
import os
import re
gGemmParamTuneHeadFile = "CutlassGemmParamTune.hpp"
gGemmBatchedParamTuneHeadFile = "CutlassGemmBatchedParamTune.hpp"

def generateGemmTuneFile(headfile):
	hpp = "#ifdef ENABLE_CUDA_TUNE_PARAM\n\n"
	hpp += "#include \"../../CutlassGemmParam.hpp\"\n\n"
	hpp += "namespace MNN {\n"
	hpp += "namespace CUDA {\n"

	data_precision = ["cutlass::half_t", "float"]
	processor_type = ["cutlass::arch::OpClassTensorOp"]
	sm_arch_type   = ["cutlass::arch::Sm80"]
	mma16816_shape  = "cutlass::gemm::GemmShape<16, 8, 16>"
	epilogue_type  = ["LinearCombination", "LinearCombinationRelu", "LinearCombinationRelu6"]
	layout_a_name = "_Row"
	layout_b_name = "_Column"
	thread_block_shapes = ["cutlass::gemm::GemmShape<64, 64, 64>", "cutlass::gemm::GemmShape<128, 64, 64>", "cutlass::gemm::GemmShape<64, 64, 32>", "cutlass::gemm::GemmShape<128, 64, 32>", "cutlass::gemm::GemmShape<64, 128, 32>", "cutlass::gemm::GemmShape<256, 64, 32>", "cutlass::gemm::GemmShape<128, 128, 32>"]
	warp_shapes         = ["cutlass::gemm::GemmShape<32, 32, 64>", "cutlass::gemm::GemmShape<64, 32, 64>",  "cutlass::gemm::GemmShape<32, 32, 32>", "cutlass::gemm::GemmShape<64, 32, 32>",  "cutlass::gemm::GemmShape<32, 64, 32>",  "cutlass::gemm::GemmShape<64, 64, 32>" ,  "cutlass::gemm::GemmShape<64, 64, 32>"]
	NumStages = "3"

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
				gemm_name = "GemmTensor"
				processor_name = ["EpilogueCudaOp", "EpilogueTensorOp"]

				for smarch in sm_arch_type:
					sm_name = "_Sm80"
					element_input_precision = precision
					for element_output_precision in data_precision:
						if element_input_precision == "float":# and element_output_precision == "cutlass::half_t":
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

							for thread_block_shape in thread_block_shapes:
								warp_shape = "cutlass::gemm::GemmShape<32, 32, 64>"
								block_size = "_64x64x64"
								if thread_block_shape == "cutlass::gemm::GemmShape<64, 64, 64>":
									warp_shape = "cutlass::gemm::GemmShape<32, 32, 64>"
									block_size = "_64x64x64"
									NumStages = "3"
								elif thread_block_shape == "cutlass::gemm::GemmShape<128, 64, 32>":
									warp_shape = "cutlass::gemm::GemmShape<64, 32, 32>"
									block_size = "_128x64x32"
									NumStages = "4"
								elif thread_block_shape == "cutlass::gemm::GemmShape<64, 128, 32>":
									warp_shape = "cutlass::gemm::GemmShape<32, 64, 32>"
									block_size = "_64x128x32"
									NumStages = "4"									
								elif thread_block_shape == "cutlass::gemm::GemmShape<256, 64, 32>":
									warp_shape = "cutlass::gemm::GemmShape<64, 64, 32>"
									block_size = "_256x64x32"
									NumStages = "3"
								elif thread_block_shape == "cutlass::gemm::GemmShape<128, 128, 32>":
									warp_shape = "cutlass::gemm::GemmShape<64, 64, 32>"
									block_size = "_128x128x32"
									NumStages = "3"
								elif thread_block_shape == "cutlass::gemm::GemmShape<64, 64, 32>":
									warp_shape = "cutlass::gemm::GemmShape<32, 32, 32>"
									block_size = "_64x64x32"
									NumStages = "6"
								elif thread_block_shape == "cutlass::gemm::GemmShape<128, 64, 64>":
									warp_shape = "cutlass::gemm::GemmShape<64, 32, 64>"
									block_size = "_128x64x64"
									NumStages = "2"

								hpp += "using " + gemm_name + inp_precision_name + out_precision_name + epilogue_name + out_align_name + layout_a_name + layout_b_name + sm_name + block_size + " = cutlass::gemm::device::Gemm<\n    "
								hpp += element_input_precision + ",\n    LayoutInputA,\n    "
								hpp += element_input_precision + ",\n    LayoutInputB,\n    "
								hpp += element_output_precision + ",\n    LayoutOutput,\n    ElementAccumulator,\n    "
								hpp += processor + ",\n    " + smarch + ",\n    "
								hpp += thread_block_shape + ",\n    " + warp_shape + ",\n    "
								hpp += mma16816_shape + ",\n    "

								hpp += out_align + out_precision_name + epilogue_name + ",\n    "
								hpp += "SwizzleThreadBlock,\n    "
								hpp += NumStages
								if sm_name == "_Sm80":
									hpp += ",\n    128 / cutlass::sizeof_bits<" + element_input_precision + ">::value, 128 / cutlass::sizeof_bits<" + element_input_precision + ">::value, true>;\n\n"
								else :
									hpp += ">;\n\n"

	hpp += "}\n}\n#endif"
	with open(headfile, "w") as f:
		f.write(hpp)

def generateGemmBatchedTuneFile(headfile):
	hpp = "#ifdef ENABLE_CUDA_TUNE_PARAM\n\n"
	hpp += "#include \"../../CutlassGemmParam.hpp\"\n"
	hpp += "#include \"cutlass/gemm/device/gemm_batched.h\"\n\n"
	hpp += "namespace MNN {\n"
	hpp += "namespace CUDA {\n"

	hpp += "using BatchedSwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;\n\n"

	thread_block_shapes = ["cutlass::gemm::GemmShape<64, 64, 64>", "cutlass::gemm::GemmShape<128, 64, 64>", "cutlass::gemm::GemmShape<64, 64, 32>", "cutlass::gemm::GemmShape<128, 64, 32>", "cutlass::gemm::GemmShape<64, 128, 32>", "cutlass::gemm::GemmShape<256, 64, 32>", "cutlass::gemm::GemmShape<128, 128, 32>"]
	warp_shapes         = ["cutlass::gemm::GemmShape<32, 32, 64>", "cutlass::gemm::GemmShape<64, 32, 64>",  "cutlass::gemm::GemmShape<32, 32, 32>", "cutlass::gemm::GemmShape<64, 32, 32>",  "cutlass::gemm::GemmShape<32, 64, 32>",  "cutlass::gemm::GemmShape<64, 64, 32>",  "cutlass::gemm::GemmShape<64, 64, 32>"]
	NumStages = "3"

	data_precision = ["cutlass::half_t", "float"]
	processor_type = ["cutlass::arch::OpClassTensorOp"]
	sm_arch_type   = ["cutlass::arch::Sm80"]
	mma16816_shape  = "cutlass::gemm::GemmShape<16, 8, 16>"
	layout_type    = ["cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor"]
	for element_input_precision in ["cutlass::half_t"]:
		inp_precision_name = "_F16"
		for processor in processor_type:
			gemm_name = "GemmBatchedTensor"
			processor_name = ["EpilogueCudaOp", "EpilogueTensorOp"]

			for smarch in sm_arch_type:
				sm_name = "_Sm80"
				for element_output_precision in ["cutlass::half_t", "float"]:
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
							
							for thread_block_shape in thread_block_shapes:
								warp_shape = "cutlass::gemm::GemmShape<32, 32, 64>"
								block_size = "_64x64x64"
								if thread_block_shape == "cutlass::gemm::GemmShape<64, 64, 64>":
									warp_shape = "cutlass::gemm::GemmShape<32, 32, 64>"
									block_size = "_64x64x64"
									NumStages = "3"
								elif thread_block_shape == "cutlass::gemm::GemmShape<128, 64, 32>":
									warp_shape = "cutlass::gemm::GemmShape<64, 32, 32>"
									block_size = "_128x64x32"
									NumStages = "4"
								elif thread_block_shape == "cutlass::gemm::GemmShape<64, 128, 32>":
									warp_shape = "cutlass::gemm::GemmShape<32, 64, 32>"
									block_size = "_64x128x32"
									NumStages = "4"								
								elif thread_block_shape == "cutlass::gemm::GemmShape<256, 64, 32>":
									warp_shape = "cutlass::gemm::GemmShape<64, 64, 32>"
									block_size = "_256x64x32"
									NumStages = "3"
								elif thread_block_shape == "cutlass::gemm::GemmShape<128, 128, 32>":
									warp_shape = "cutlass::gemm::GemmShape<64, 64, 32>"
									block_size = "_128x128x32"
									NumStages = "3"
								elif thread_block_shape == "cutlass::gemm::GemmShape<64, 64, 32>":
									warp_shape = "cutlass::gemm::GemmShape<32, 32, 32>"
									block_size = "_64x64x32"
									NumStages = "6"
								elif thread_block_shape == "cutlass::gemm::GemmShape<128, 64, 64>":
									warp_shape = "cutlass::gemm::GemmShape<64, 32, 64>"
									block_size = "_128x64x64"
									NumStages = "2"

								hpp += "using " + gemm_name + inp_precision_name + out_precision_name + "Linear" + out_align_name + layout_a_name + layout_b_name + sm_name + block_size + " = cutlass::gemm::device::GemmBatched<\n    "
								hpp += element_input_precision + ",\n    " + layout_a + ",\n    "
								hpp += element_input_precision + ",\n    " + layout_b + ",\n    "
								hpp += element_output_precision + ",\n    LayoutOutput,\n    ElementAccumulator,\n    "
								hpp += processor + ",\n    " + smarch + ",\n    "
								hpp += thread_block_shape + ",\n    " + warp_shape + ",\n    "
								hpp += mma16816_shape + ",\n    "

								hpp += out_align + out_precision_name + "Linear,\n    "
								hpp += "BatchedSwizzleThreadBlock,\n    "
								hpp += NumStages + ">;\n\n"

	hpp += "}\n}\n#endif"
	with open(headfile, "w") as f:
		f.write(hpp)

if __name__ == '__main__':
    generateGemmTuneFile(gGemmParamTuneHeadFile);
    generateGemmBatchedTuneFile(gGemmBatchedParamTuneHeadFile);

