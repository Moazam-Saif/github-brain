import deeplake, os, json
from dotenv import load_dotenv
load_dotenv()

org   = os.getenv("ACTIVELOOP_ORG")
token = os.getenv("ACTIVELOOP_TOKEN")
ds    = deeplake.load(f"hub://{org}/github_brain_v5", token=token)

print(f"Total chunks: {len(ds)}")
print()

for i in range(len(ds)):
    raw = ds.metadata[i].numpy()
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    meta = json.loads(str(raw))
    print(f"{i:2d} | {meta.get('file_path', 'unknown')} | chunk {meta.get('chunk_index')}")