import subprocess, numpy as np, json, pathlib
W,H=108,192
def fr(p):
    r=subprocess.run(["ffmpeg","-v","error","-i",str(p),"-vf",f"scale={W}:{H}","-pix_fmt","gray",
                      "-f","rawvideo","-"],capture_output=True).stdout
    return np.frombuffer(r,np.uint8).reshape(-1,H,W).astype(np.int16)
new=fr("final/FINAL_lipsynced_all.mp4"); old=fr(next(pathlib.Path("source").glob("*.mp4")))
n=min(len(new),len(old)); print("frames:",len(new),len(old))
d=np.abs(new[:n]-old[:n]).mean(axis=(1,2))
plan={s["id"]:s for s in json.load(open("metadata/segments.json"))}
segs=[(plan[s]["start_frame"],plan[s]["end_frame"],s) for s in json.load(open("metadata/usable.json"))]
segs.append((3100,3209,"seg001")); segs.sort()
inside=np.zeros(n,bool)
for a,b,_ in segs: inside[a:b+1]=True
out=d[~inside]
print(f"OUTSIDE segments: mean={out.mean():.3f} max={out.max():.2f}  (should be ~0)")
bad=np.where((~inside)&(d>3))[0]
print(f"misaligned frames outside segments: {len(bad)}")
if len(bad): print("  ranges:", bad[:20], "...")
for a,b,sid in segs:
    print(f"  {sid:8} {a}-{b}  inside mean={d[a:b+1].mean():.2f}")
