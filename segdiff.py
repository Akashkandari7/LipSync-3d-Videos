import subprocess, numpy as np, json, pathlib
W,H=108,192
def fr(p):
    r=subprocess.run(["ffmpeg","-v","error","-i",str(p),"-vf",f"scale={W}:{H}","-pix_fmt","gray",
                      "-f","rawvideo","-"],capture_output=True).stdout
    return np.frombuffer(r,np.uint8).reshape(-1,H,W).astype(np.int16)
plan={s["id"]:s for s in json.load(open("metadata/segments.json"))}
ids=json.load(open("metadata/usable.json"))+["seg001"]
print(f"{'seg':8}{'frames':>8}{'AI change (mean abs diff vs input)':>36}")
for sid in sorted(ids):
    i=pathlib.Path("segments/input")/sid/("seg.mp4" if sid!="seg001" else "seg001.mp4")
    o=pathlib.Path("segments/output")/sid/"synced.mp4"
    a,b=fr(i),fr(o)
    n=min(len(a),len(b))
    d=np.abs(a[:n]-b[:n]).mean()
    print(f"{sid:8}{n:8d}{d:>30.2f}")
