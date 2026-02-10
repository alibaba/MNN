import sys
import re

class Assembly():
    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path
        # instructions
        self.ops = ['sdot', 'udot', 'smmla', 'bfmmla', 'mov', 'smopa', 'fmopa', 'luti4', 'ldr']

    def assembly(self):
        self.dst_content = []
        src = open(self.src_path, 'rt')
        for line in src.readlines():
            code = line
            cmd = code.strip().split(' ')
            for op in self.ops:
                if cmd[0] == op:
                    if op == 'mov':
                        code = getattr(self, op)(code, cmd[1], cmd[2])
                    elif op == 'smopa' or op == 'fmopa' or op == 'luti4':
                        inst = getattr(self, op)(code)
                        code = code[:code.find(op)] + inst + ' // ' + code.strip(' ')
                    elif op == 'ldr':
                        if cmd[1] != 'zt0,':
                            continue
                        inst = getattr(self, op)(code)
                        code = code[:code.find(op)] + inst + ' // ' + code.strip(' ')
                    else:
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

    def udot(self, operand1, operand2, operand3):
        # UDOT <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tc>[offset]
        Vd, Ta = self.operand_spilt(operand1)
        Vn, Tb = self.operand_spilt(operand2)
        Vm, Tc = self.operand_spilt(operand3)
        if "[" in Tc:
            # other flag:
            # offset = flag[4] * 2 + opcode[-1]
            # dst == '4s' ? opcode[1] = 1 : opcode[1] = 0
            Tc, offset = self.t_split(Tc)
            opcode = list('01101111100')
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
            opcode = list('01101110100') # different from the case with offset.
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

    def mov(self, code, operand1, operand2):
        # compile failed using `mov v1.8h, v2.8h`
        # change to `mov v1.16b, v2.16b`
        if '.8h' not in operand1 or '.8h' not in operand2:
            return code
        operand1 = operand1.replace('8h', '16b')
        operand2 = operand2.replace('8h', '16b')
        new_mov = f'mov {operand1} {operand2}'
        new_code = code[:code.find('mov')] + new_mov + ' // ' + code.strip(' ')
        return new_code

    def smopa(self, instruction):
        """
        SMOPA <ZAda>.S, <Pn>/M, <Pm>/M, <Zn>.B, <Zm>.B 32bit 4-way
        SMOPA <ZAda>.D, <Pn>/M, <Pm>/M, <Zn>.H, <Zm>.H 64bit 4-way
        """
        try:
            parts = instruction.replace(' ', '').split(',')
            if len(parts) != 5:
                raise ValueError("smopa 指令格式错误")

            zda = int(parts[0].split('za')[1].split('.')[0])
            pn = int(parts[1].split('p')[1].split('/')[0])
            pm = int(parts[2].split('p')[1].split('/')[0])
            zn = int(parts[3].split('z')[1].split('.')[0])
            zm = int(parts[4].split('z')[1].split('.')[0])

            zmDataType = parts[4].split('z')[1].split('.')[1][0]

            if not (0 <= zda <= 15):
                raise ValueError("zda必须在0-15范围内")
            if not (0 <= pn <= 7):
                raise ValueError("pg必须在0-7范围内")
            if not (0 <= pn <= 7):
                raise ValueError("pn必须在0-7范围内")
            if not (0 <= zm <= 31):
                raise ValueError("zm必须在0-31范围内")
            if not (0 <= zn <= 31):
                raise ValueError("zn必须在0-31范围内")

            # smopa za0.s, p3/m, p4/m, z0.b, z1.b
            is32Bit4way = (parts[0].split('za')[1].split('.')[1] == "s") and (parts[3].split('z')[1].split('.')[1] == 'b') and (zmDataType == 'b')
            # smopa za0.d, p3/m, p4/m, z0.h, z1.h
            is64Bit4way = (parts[0].split('za')[1].split('.')[1] == "d") and (parts[3].split('z')[1].split('.')[1] == 'h') and (zmDataType == 'h')
            # smopa za0.s, p3/m, p4/m, z0.h, z1.h
            is2way = (parts[0].split('za')[1].split('.')[1] == "s") and (parts[3].split('z')[1].split('.')[1] == 'h') and (zmDataType == 'h')
            if (is32Bit4way == False) and (is64Bit4way == False) and (is2way):
                raise ValueError("smopa 指令格式错误")

            # is32Bit4way
            opcode = "10100000100"     #[31, 21]
            zmCode = format(zm, '05b') # zm register has '5' bit,[20, 16]
            pmCode = format(pm, '03b') # pm register has '3' bit,[15,13]
            pnCode = format(pn, '03b') # pn register has '3' bit,[12,10]
            znCode = format(zn, '05b') # zn register has '5' bit, [9,5]
            fixCode = "000"            # fixed encode
            zaCode = format(zda, '02b') # za register has '2' bit, [1,0]

            if is64Bit4way == True:
                opcode = "10100000110"
                fixCode = "00"
                zaCode = format(zda, '03b')
            elif is2way == True:
                opcode = "10100000100"
                fixCode = "010"

            # concact
            binary = opcode + zmCode + pmCode + pnCode + znCode + fixCode + zaCode
            inst = '.inst ' + str(hex(int(binary, 2)))
            return inst

        except Exception as e:
            raise ValueError(f"smopa 指令解析错误: {str(e)}")

    def fmopa(self, instruction):
        '''
        FMOPA <ZAda>.S, <Pn>/M, <Pm>/M, <Zn>.S, <Zm>.S
        '''
        try:
            parts = instruction.replace(' ', '').split(',')
            if len(parts) != 5:
                raise ValueError("fmopa 指令格式错误")

            zda = int(parts[0].split('za')[1].split('.')[0])
            pn = int(parts[1].split('p')[1].split('/')[0])
            pm = int(parts[2].split('p')[1].split('/')[0])
            zn = int(parts[3].split('z')[1].split('.')[0])
            zm = int(parts[4].split('z')[1].split('.')[0])

            zmDataType = parts[4].split('z')[1].split('.')[1][0]

            # fmopa za0.s, p3/m, p4/m, z0.s, z1.s
            singlePrecisionNotWidening = (parts[0].split('za')[1].split('.')[1] == "s") and (parts[3].split('z')[1].split('.')[1] == 's') and (zmDataType == 's')
            # fmopa za0.s, p3/m, p3/m, z0.h, z1.h
            fp16Tofp32 = (parts[0].split('za')[1].split('.')[1] == "s") and (parts[3].split('z')[1].split('.')[1] == 'h') and (zmDataType == 'h')

            if not singlePrecisionNotWidening and not fp16Tofp32:
                raise ValueError("Not implement yet\n")

            opcode = "10000000100"
            zmCode = format(zm, '05b') # zm register has '5' bit,[20, 16]
            pmCode = format(pm, '03b') # pm register has '3' bit,[15,13]
            pnCode = format(pn, '03b') # pn register has '3' bit,[12,10]
            znCode = format(zn, '05b') # zn register has '5' bit, [9,5]
            fixCode = "000"            # fixed encode
            zaCode = format(zda, '02b') # za register has '2' bit, [1,0]

            if fp16Tofp32 == True:
                opcode = "10000001101"

            binary = opcode + zmCode + pmCode + pnCode + znCode + fixCode + zaCode
            inst = '.inst ' + str(hex(int(binary, 2)))
            return inst

        except Exception as e:
            raise ValueError(f"fmopa 指令解析错误: {str(e)}")

    def luti4(self, instruction):
        '''
        luti4 {z2.b-z3.b}, zt0, z1[0]
        '''
        try:
            parts = instruction.replace(' ', '').split(',')
            if len(parts) != 3:
                raise ValueError("luti4 指令格式错误")

            # 解析目标寄存器
            zd = int(parts[0].split('z')[1].split('.')[0])
            T = parts[0].split('.')[1].split('.')[0][0]
            if T != 'b':
                raise ValueError("Not implement yet\n")

            # 解析查找表寄存器
            zt = int(parts[1].split('zt')[1])

            # 解析源寄存器
            zn = int(parts[2].split('z')[1].split('[')[0])
            i2 = int(parts[2].split('z')[1].split('[')[1][0])

            opcode = "110000001000101"
            i2code = format(i2, '02b')
            constcode0 = "1"
            sizecode = "00" # b
            constcode1 = "00"
            zncode = format(zn, '05b')
            zdcode = format(zd, '05b')

            binary = opcode + i2code + constcode0 + sizecode + constcode1 + zncode + zdcode
            inst = '.inst ' + str(hex(int(binary, 2)))
            return inst

        except Exception as e:
            raise ValueError(f"luti4 指令解析错误: {str(e)}")

    def ldr(self, instruction):
        '''
        .inst 0xe11f8100  // ldr zt0, [x8]
        '''
        i0 = instruction.find('[')
        i1 = instruction.find(']')
        x = int(instruction[i0 + 2: i1])
        opcode = "1110000100011111100000"
        rn = format(x, '05b')
        fixcode = "00000"
        binary = opcode + rn + fixcode
        inst = '.inst ' + str(hex(int(binary, 2)))
        return inst


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
