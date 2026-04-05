import torch
from transformers import AutoTokenizer, AutoModel

print("=== SYSTEM CHECK ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

print("\n=== GPU INFO ===")
print("GPU name:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))

device = torch.device("cuda")

print("\n=== BASIC CUDA TENSOR TEST ===")
x = torch.randn(1024, 1024, device=device)
y = torch.matmul(x, x)
torch.cuda.synchronize()
print("CUDA tensor ops: OK")

print("\n=== TRANSFORMERS MODEL TEST ===")
model_name = "GSAI-ML/LLaDA-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

text = "GPU sanity check"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    out = model(**inputs)

torch.cuda.synchronize()
print("Model forward pass: OK")

print("\n=== MEMORY CHECK ===")
allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
print(f"Allocated: {allocated:.1f} MB")
print(f"Reserved : {reserved:.1f} MB")

print("\nALL GPU TESTS PASSED ✅")