import sys

class Assembly():
    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path
        # instructions
        self.ops = ['sdot', 'smmla', 'bfmmla']
    def assembly(self):
        self.dst_content = []
        src = open(self.src_path, 'rt')
        for line in src.readlines():
            code = line
            cmd = code.strip().split(' ')
            for op in self.ops:
                if cmd[0] == op:
                    inst = getattr(self, op)(cmd[1], cmd[2], cmd[3])
                    code = code[:code.find(op)] + inst + ' // ' + code.strip(' ')
            self.dst_content.append(code)
        src.close()
        self.write()
    def write(self):
        dst = open(self.dst_path, 'wt')
        dst.writelines(self.dst_content)
        dst.close()
    # asm parse helper function
    def gen_inst(self, opcode, flag, r1, r2, r3):
        cmd = opcode + r1 + flag + r2 + r3
        inst = '.inst ' + str(hex(int(cmd, 2)))
        return inst
    def register_to_bin(self, register):
        assert(register[0] == 'v')
        id = str(bin(int(register[1:])))[2:]
        id = '0' * (5 - len(id)) + id
        return id
    def operand_spilt(self, operand):
        v, t = operand.split('.')
        return self.register_to_bin(v), t
    def operand_to_bin(self, operand):
        r, _ = self.operand_spilt(operand)
        return r
    def t_split(self, t):
        idx = None
        if t[-1] == ']':
            t, offset = t[:-1].split('[')
        return t, int(offset)
    # instruction code gen function
    def sdot(self, operand1, operand2, operand3):
        # SDOT <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tc>[offset]
        Vd, Ta = self.operand_spilt(operand1)
        Vn, Tb = self.operand_spilt(operand2)
        Vm, Tc = self.operand_spilt(operand3)
        if "[" in Tc:
            # other flag:
            # offset = flag[4] * 2 + opcode[-1]
            # dst == '4s' ? opcode[1] = 1 : opcode[1] = 0
            Tc, offset = self.t_split(Tc)
            opcode = list('01001111100')
            flag = list('111000')
            # set Q
            if Ta == '2s' and Tb == '8b':
                opcode[1] = '0'
            # set offset
            if offset == 1 or offset == 3:
                opcode[-1] = '1'
            if offset == 2 or offset == 3:
                flag[4] = '1'
            opcode = ''.join(opcode)
            flag = ''.join(flag)
            return self.gen_inst(opcode, flag, Vm, Vn, Vd)
        else:
            opcode = list('01001110100') # different from the case with offset.
            flag   = list('100101')
            # set Q
            if "2s" in Ta and "8b" in Tb:
                opcode[1] = '0'
            
            opcode = ''.join(opcode)
            flag = ''.join(flag)
            return self.gen_inst(opcode, flag, Vm, Vn, Vd)

    def smmla(self, operand1, operand2, operand3):
        # SMMLA <Vd>.4S, <Vn>.16B, <Vm>.16B
        opcode = '01001110100'
        flag = '101001'
        Vd = self.operand_to_bin(operand1)
        Vn = self.operand_to_bin(operand2)
        Vm = self.operand_to_bin(operand3)
        return self.gen_inst(opcode, flag, Vm, Vn, Vd)
    def bfmmla(self, operand1, operand2, operand3):
        # BFMMLA <Vd>.4S, <Vn>.8H, <Vm>.8H
        opcode = '01101110010'
        flag = '111011'
        Vd = self.operand_to_bin(operand1)
        Vn = self.operand_to_bin(operand2)
        Vm = self.operand_to_bin(operand3)
        return self.gen_inst(opcode, flag, Vm, Vn, Vd)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python arm_asselmbly.py src.asm [dst.asm]')
    src_file = sys.argv[1]
    if len(sys.argv) > 2:
        dst_file = sys.argv[2]
    else:
        dst_file = src_file
    a = Assembly(src_file, dst_file)
    a.assembly()
