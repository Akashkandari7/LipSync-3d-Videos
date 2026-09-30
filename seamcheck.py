import subprocess, numpy as np, json
W,H=108,192
def diffs(p):
    r=subprocess.run(["ffmpeg","-v","error","-i",p,"-vf",f"scale={W}:{H}","-pix_fmt","gray",
                      "-f","rawvideo","-"],capture_output=True).stdout
    a=np.frombuffer(r,np.uint8).reshape(-1,H,W).astype(np.int16)
    return np.abs(np.diff(a,axis=0)).mean(axis=(1,2))
new=diffs("final/FINAL_lipsynced_all.mp4"); old=diffs(next(__import__('pathlib').Path("source").glob("*.mp4")).as_posix())
plan={s["id"]:s for s in json.load(open("metadata/segments.json"))}
segs=[(plan[s]["start_frame"],plan[s]["end_frame"],s) for s in json.load(open("metadata/usable.json"))]
segs.append((3100,3209,"seg001")); segs.sort()
print(f"baseline frame-to-frame diff: original mean={old.mean():.2f}  final mean={new.mean():.2f}")
print(f"\n{'seg':8}{'boundary':>12}{'final':>8}{'orig':>8}{'delta':>8}  flag")
worst=0
for a,b,sid in segs:
    for lbl,i in ((f"in @{a}",a-1),(f"out @{b+1}",b)):
        if 0<=i<len(new):
            d=new[i]-old[i]; worst=max(worst,abs(d))
            flag="" if abs(d)<old.std() else "CHECK"
            print(f"{sid:8}{lbl:>12}{new[i]:8.2f}{old[i]:8.2f}{d:+8.2f}  {flag}")
print(f"\nlargest seam delta: {worst:.2f}   (original std dev: {old.std():.2f})")
