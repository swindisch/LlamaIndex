# import torch
# x = torch.rand(5, 3)
# print(x)

# print(f'{torch.cuda.is_available()}')
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))


from llama_cpp import Llama
llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf", n_gpu_layers=30, n_ctx=3584, n_batch=521, verbose=True)
# adjust n_gpu_layers as per your GPU and model
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)