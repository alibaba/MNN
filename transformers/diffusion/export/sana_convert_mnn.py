import os
def convert(onnx_path, mnn_path, extra):
    print('Onnx path: ', onnx_path)
    print('MNN path: ', mnn_path)
    print('Extra: ', extra)
    convert_path = '../../../build/MNNConvert'
    if not os.path.exists(convert_path):
        print(convert_path + " not exist, use pymnn instead")
        convert_path = 'mnnconvert'
    models = ['connector', 'projector', 'transformer', 'vae_encoder', 'vae_decoder']
    for model in models:
        cmd = convert_path + ' -f ONNX --modelFile ' + onnx_path + "/" + model + '.onnx --MNNModel ' + os.path.join(mnn_path, model + '.mnn') + ' --saveExternalData=1 --weightQuantBits=8 ' + extra
        print(cmd)
        print(os.popen(cmd).read())

if __name__ == '__main__':
    import sys
    onnx_dir = sys.argv[1]
    llm_dir = sys.argv[2]
    dst_dir = sys.argv[3]
    extra = ""
    extra = " ".join(sys.argv[4:])
    # convert diffusion model
    convert(onnx_dir, dst_dir, extra)
    import subprocess, sys
    from pathlib import Path
    this_dir = Path(__file__).resolve().parent
    llmexport = (this_dir / "../../llm/export/llmexport.py").resolve()
    # convert llm model
    subprocess.run([
        sys.executable,
        str(llmexport),
        "--path", llm_dir,
        "--export", "mnn",
        "--dst_path", dst_dir + "/llm",
    ], check=True)
    
    import torch
    meta_queries_path = os.path.join(onnx_dir, 'meta_queries.pt')
    meta_queries = torch.load(meta_queries_path)
    print(f"✓ Meta Queries Loaded: {meta_queries.shape}")
        
    # convert meta_queries
    import MNN.expr as expr
    torch_meta_queries = meta_queries.float().contiguous().cpu()
    mnn_meta_queries = expr.const(torch_meta_queries.data_ptr(), torch_meta_queries.shape, expr.data_format.NCHW, expr.dtype.float)
    mnn_meta_queries.name = 'meta_queries'
    expr.save([mnn_meta_queries], dst_dir + f'/llm/meta_queries.mnn')
