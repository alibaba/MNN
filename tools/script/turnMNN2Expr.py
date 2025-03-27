import json
import copy
def check_same_op(src, dst):
    if src['type'] != dst['type']:
        return False
    if 'main' in src:
        if 'main' not in dst:
            return False
        else:
            return src['main'] == dst['main']
    if 'main' in dst:
        return False
    return True

class Function:
    def __init__(self, type, op, input_number, output_number):
        self.name = None
        self.type = type
        self.op = op
        self.input_number = input_number
        self.output_number = output_number

def op_to_function(op):
    num_inputs = 0
    op = copy.deepcopy(op)
    if 'inputIndexes' in op:
        num_inputs = len(op['inputIndexes'])
        del op['inputIndexes']
    num_outputs = len(op['outputIndexes'])
    del op['outputIndexes']
    if 'name' in op:
        del op['name']
    optype = op['type']
    op = json.dumps(op, indent=4)
    return Function(optype, op, num_inputs, num_outputs)

def remove_parameters(op):
    if 'main' not in op:
        return
    main_p = op['main']
    if op['main_type'] == 'Convolution2D':
        if 'weight' in main_p:
            del main_p['weight']
        if 'bias' in main_p:
            del main_p['bias']
        if 'external' in main_p:
            del main_p['external']
        if 'quanParameter' in main_p:
            del main_p['quanParameter']
    elif op['main_type'] == 'LayerNorm':
        if 'gamma' in main_p:
            del main_p['gamma']
        if 'beta' in main_p:
            del main_p['beta']

class MNNInfo:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.origin = None
        self.sep_functions = None
        self.function_list = None
        self.consts = None
        self.tensor_name = None

def load_mnn(filename, remove_parameter = True):
    dst = MNNInfo()
    mnn = {}
    function_list = {}
    sep_functions = []
    inputs = []
    consts = []
    with open(filename) as f:
        mnn = json.load(f)
        oplists = mnn['oplists']
        if remove_parameter:
            for op in oplists:
                remove_parameters(op)
        for i in range(0, len(oplists)):
            seperate = True
            op = oplists[i]
            if 'inputIndexes' not in op or len(op["inputIndexes"]) == 0:
                if op['type'] == 'Input':
                    inputs.append(op)
                    continue
            for j in range(0, i):
                if check_same_op(op, oplists[j]):
                    function_list[i] = function_list[j]
                    seperate = False
                    break
            if seperate:
                function_list[i] = op_to_function(op)
                function_list[i].name = function_list[i].type + '%d' %i
                sep_functions.append(function_list[i])
    dst.origin = mnn
    dst.sep_functions = sep_functions
    dst.function_list = function_list
    dst.inputs = inputs
    dst.consts = consts
    dst.tensor_name = {}
    tensorName = mnn['tensorName']
    for i in range(0, len(tensorName)):
        dst.tensor_name[tensorName[i]] = i
    if 'outputName' in mnn:
        dst.outputs = mnn['outputName']
    else:
        mask = {}
        for i in range(0, len(tensorName)):
            mask[i] = 0
        # Find not middle as output
        dst.outputs = []
        oplists = mnn['oplists']
        for op in oplists:
            if 'inputIndexes' in op:
                for index in op['inputIndexes']:
                    mask[index] |= 1
            for index in op['outputIndexes']:
                mask[index] |= 2
        for i in range(0, len(tensorName)):
            if mask[i] == 2:
                dst.outputs.append(tensorName[i])        
    return dst

def make_python(funcname, mnninfo):
    mnn = mnninfo.origin
    sep_functions = mnninfo.sep_functions
    function_list = mnninfo.function_list
    tensor_names = mnn['tensorName']
    inputs = mnninfo.inputs
    inputnames = list(map(lambda op:tensor_names[op['outputIndexes'][0]], inputs))
    inputindexes = list(map(lambda op:op['outputIndexes'][0], inputs))
    main_str = "import MNN.expr as F\n"
    main_str += 'def ' + funcname + '(' + inputnames[0]
    for i in range(1, len(inputnames)):
        main_str += ', ' + inputnames[i]
    main_str += '):\n'
    indent = "    "
    for func in sep_functions:
        main_str += indent + func.name + ' = \"\"\"' + func.op
        main_str += indent + '\"\"\"\n'
    main_str += indent + 'stackes = {}\n'
    for i in range(0, len(inputindexes)):
        main_str += indent + 'stackes[%d]' %inputindexes[i] + ' = ' + inputnames[i] + '\n'
    main_str += indent + '# Call Funciton\n'
    oplists = mnn['oplists']
    for i in range(0, len(oplists)):
        op = oplists[i]
        if op['type'] == 'Input':
            continue
        main_str += indent + 'p = F.jsonop(['
        if 'inputIndexes' in op and len(op["inputIndexes"]) > 0:
            main_str += 'stackes[%d]' %op["inputIndexes"][0]
            for j in range(1, len(op['inputIndexes'])):
                main_str += ', stackes[%d]' %op["inputIndexes"][j]
        main_str += '], ' + function_list[i].name  + ', %d)\n' %len(op['outputIndexes'])
        main_str += indent + 'stackes[%d]' %op['outputIndexes'][0]
        for j in range(1, len(op['outputIndexes'])):
            main_str += ', stackes[%d]' %op['outputIndexes'][j]
        main_str += ' = p[0]'
        for j in range(1, len(op['outputIndexes'])):
            main_str += ', p[%d]' %j
        main_str += '\n'


    # Return output
    outputs = mnninfo.outputs
    for output in outputs:
        index = mnninfo.tensor_name[output]
        main_str += indent + 'stackes[%d]' %index +  '.name = \''  + output + '\'\n'
    main_str += indent + 'return stackes[%d]' %mnninfo.tensor_name[outputs[0]]
    for i in range(1, len(outputs)):
        main_str += ', stackes[%d]' %mnninfo.tensor_name[outputs[i]]
    main_str +='\n'

    print("Write to python ", funcname + '.py')
    with open(funcname + '.py', 'w') as f:
        f.write(main_str)
    return

def make_cpp(cppname, mnninfo):
    mnn = mnninfo.origin
    sep_functions = mnninfo.sep_functions
    function_list = mnninfo.function_list
    inputs = mnninfo.inputs
    outputs = mnninfo.outputs
    consts = mnninfo.consts

    hpp_file_name = cppname + '.hpp'
    cpp_file_name = cppname + '.cpp'

    cppstr = "#include <MNN/expr/ExprCreator.hpp>\n"
    cppstr += "#include \"" + hpp_file_name + "\"\n"
    # Init Sub function
    for i in range(len(sep_functions)):
        func = sep_functions[i]
        cppstr += 'static std::vector<VARP> ' + func.name + '(VARP x0'
        for j in range(1, func.input_number):
            cppstr += ', VARP x%d' %j
        cppstr += ') {\n'
        cppstr += "static const char* " + func.name + "_main" + " = R\"func(\n"
        cppstr += func.op
        cppstr += ")func\";\n"
        cppstr += 'return _JSONOp({x0'
        for j in range(1, func.input_number):
            cppstr += ', x%d' %j
        cppstr += '}, '
        cppstr += func.name + "_main"
        cppstr += ', %d' %func.output_number
        cppstr += ');\n'
        cppstr += '}\n'

    oplists = mnn['oplists']
    function_name = cppname.replace('/', '_').replace('\\', '_')
    tensor_names = mnn['tensorName']
    # Init input
    cppstr += 'std::map<std::string, VARP> ' + function_name + '(const std::map<std::string, VARP>& ____inputs) {\n'
    cppstr += 'std::vector<VARP> t(%d);\n' %len(tensor_names)
    cppstr += '// Init Inputs\n'
    for op in inputs:
        name = tensor_names[op['outputIndexes'][0]]
        cppstr += 't[%d]' %op['outputIndexes'][0] + ' = ____inputs.find(\"' + name + '\")->second;\n'

    # Call function
    cppstr += '// Call Funciton\n'
    for i in range(0, len(oplists)):
        op = oplists[i]
        if 'inputIndexes' not in op or len(op["inputIndexes"]) == 0:
            continue
        cppstr +='\n'
        cppstr += '{\n'
        cppstr += "VARPS tmp = " + function_list[i].name
        if 'inputIndexes' in op:
            cppstr += '(t[%d]' %op['inputIndexes'][0]
            for v in range(1, len(op['inputIndexes'])):
                cppstr += ',t[%d]' %op['inputIndexes'][v]
            cppstr +=');\n'
        else:
            cppstr += '();\n'
        for i in range(len(op['outputIndexes'])):
            index = op['outputIndexes'][i]
            cppstr += 't[%d] = ' %index + 'tmp[%d];\n' %i
        cppstr += '}\n'
    # Generate outputs
    cppstr += '// Collect Outputs\n'
    cppstr += 'std::map<std::string, VARP> _____outputs;\n'
    for output in outputs:
        index = mnninfo.tensor_name[output]
        cppstr += 't[%d] ' %index + '->setName(\"' + output+'\");\n'
        cppstr += '_____outputs[\"' + output + '\"] = t[%d];\n' %index
    cppstr += 'return _____outputs;\n'
    cppstr += '}\n'
    cppstr = cppstr.replace("VARP", "MNN::Express::VARP")
    print("Write to ", hpp_file_name, cpp_file_name)
    with open(cpp_file_name, 'w') as f:
        f.write(cppstr)
    with open(hpp_file_name, 'w') as f:
        f.write("#ifndef " + hpp_file_name.replace('.', '_')+ '\n')
        f.write("#define " + hpp_file_name.replace('.', '_') + '\n')
        f.write("#include <MNN/expr/Expr.hpp>\n")
        f.write("std::map<std::string, MNN::Express::VARP> " + function_name + '(const std::map<std::string, MNN::Express::VARP>& ____inputs);\n')
        f.write("#endif\n")

if __name__ == '__main__':
    import sys
    remove_weight = False
    if len(sys.argv) > 3:
        remove_weight = True
    mnninfo = load_mnn(sys.argv[1], remove_weight)
    make_python(sys.argv[2], mnninfo)
    make_cpp(sys.argv[2], mnninfo)

