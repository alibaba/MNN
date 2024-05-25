#!/usr/bin/python
import os
import sys


class CodeGenerator():
	"""
	compute shader define option:(assume only output)
		layout:0 ==> output
		layout:1 ==> input0
		layout:2 ==> input1
		...
		layout:n ==> userData
		layout:n+1 ==> opParam
	# param argv
		op_type, input_size, op_gpu_param
	"""
	def __init__(self, op_type, input_size, op_gpu_param, path='./Operator/'):
		self.opType = op_type
		self.cpp_class = "Vulkan" + self.opType
		self.header_file_name = path + self.cpp_class + '.hpp'
		self.cpp_file_name = path + self.cpp_class + '.cpp'
		self.input_size = input_size
		self.op_gpu_param = op_gpu_param

	def run(self):
		header = self.genHeader()
		head_fid = open(self.header_file_name, 'w')
		head_fid.write(header)
		head_fid.close()
		cpp_code = self.genCpp()
		cpp_fid = open(self.cpp_file_name, 'w')
		cpp_fid.write(cpp_code)
		cpp_fid.close()
		return

	def genHeader(self):

		res = '//\n//  xxxxx.hpp\n//  MNN\n//\n//\n' + \
			'#ifndef xxxxx_hpp\n#define xxxxx_hpp\n' + \
			'#include <MNN/backend/vulkan/execution/VulkanBasicExecution.hpp>\n\n' + \
			'namespace MNN {\n\tclass xxxxx : public VulkanBasicExecution\n\t{\n\tpublic:\n' + \
			'\t\txxxxx(const Op* op, Backend* bn);\n\t\tvirtual ~ xxxxx();\n' + \
			'\t\tErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const VulkanCommandPool::Buffer* cmdBuffer) override;\n' + \
			'\n\n\tprivate:\n' + \
			'\t\tstd::shared_ptr<VulkanBuffer> mParamBuffer;\n' + \
			'\t\tconst VulkanPipeline* mxxxxxPipeline;\n' + \
			'\t\tstd::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;\n' + \
			'\t};\n}\n#endif'

		return res.replace('xxxxx', self.cpp_class)

	def genCpp(self):

		res = '//\n//  xxxxx.cpp\n//  MNN\n//\n//\n//\n' + \
			'#include <MNN/backend/vulkan/execution/xxxxx.hpp>\n' + \
			'#include <MNN/backend/vulkan/vulkan/AllShader.h>\n#include "Macro.h"\n\n\n' + \
			'namespace MNN {\n' + \
			"\tstruct GpuParam {\n"

		for item in self.op_gpu_param:
			res += ('\t  ivec4 ' + item + ';\n')
		res += "\t};\n\n"
		# class construct function
		res += '\txxxxx::xxxxx(const Op* op, Backend* bn):VulkanBasicExecution(bn)\n\t{\n' + \
			'\t\tstd::vector<VkDescriptorType> xxxxxTypes {\n'

		for i in range(self.input_size + 1):
			res += "\t\t\tVK_DESCRIPTOR_TYPE_STORAGE_BUFFER,\n"
		res += '\t\t\tVK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,\n\t\t};\n' + \
			'\t\tauto extra = static_cast<VulkanBackend*>(bn);\n' + \
			'\t\tmxxxxxPipeline = extra->getPipeline("glsl_xxxxx_comp", glsl_xxxxx_comp, glsl_xxxxx_comp_len, xxxxxTypes);\n' + \
			'\t\tmParamBuffer.reset(new VulkanBuffer(extra->getContext(), sizeof(GpuParam), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));\n\t}\n' + \
			'\txxxxx::~xxxxx()\n\t{\n\n\t}\n'

		res += '\tErrorCode xxxxx::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const VulkanCommandPool::Buffer* cmdBuffer)\n\t{\n' + \
			'\t\tauto xxxxxParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());\n' + \
			'\t\t::memset(xxxxxParam, 0, sizeof(GpuParam));\n\n' + \
			'\t\t***put your own code here***\n\n\n' + \
			'\t\tmParamBuffer->flush(true, 0, sizeof(GpuParam));\n' + \
			'\t\tmParamBuffer->unmap();\n\n' + \
			'\t\tmDescriptorSet.reset(mxxxxxPipeline->createSet());\n' + \
			'\t\tmDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(output->deviceId()), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output->size());\n' + \
			'\t\treturn NO_ERROR;\n\t}\n\n' + \
			'\tclass xxxxxCreator : public VulkanBackend::Creator\n\t{\n\tpublic:\n' + \
			'\t\tvirtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override\n\t\t{\n' + \
			'\t\t\treturn new xxxxx(op, bn);\n\t\t}\n\t};\n\n' + \
			'\tstatic bool gResistor = [](){\n' + \
			'\t\tVulkanBackend::addCreator(OpType_xxx, new xxxxxCreator);\n\t\treturn true;\n\t}();\n\n}'

		return res.replace('xxxxx', self.cpp_class)


if __name__ == '__main__':
	print("Generate Code ...")
	print("-" * 20 + '>' * 10)
	op_type = sys.argv[1]
	input_size = int(sys.argv[2])
	op_params = ''
	if len(sys.argv) >= 4:
		op_params = sys.argv[3].split(',')
	app = CodeGenerator(op_type, input_size, op_params)
	app.run()
