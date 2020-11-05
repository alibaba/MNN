
import MNN.expr as F
import sys
modelFile = sys.argv[1]
name = sys.argv[2]
varMaps = F.load_as_dict(modelFile)

F.save([varMaps[name]], "temp.bin")
