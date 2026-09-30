import subprocess, numpy as np, json, pathlib
W,H=108,192
def rgb(p,n=None):
    r=subprocess.run(["ffmpeg","-v","error","-i",str(p),"-fps_mode","passthrough",
        "-vf",f"scale={W}:{H}","-pix_fmt","rgb24","-f","rawvideo","-"],capture_output=True).stdout
    return np.frombuffer(r,np.uint8).reshape(-1,H,W,3).astype(np.float32)
pl=json.load(open('work2/plan_refined.json'))
print(f"{'seg':6}{'spk':8}{'luma in':>9}{'luma out':>10}{'delta':>8}{'R':>7}{'G':>7}{'B':>7}{'std ratio':>11}")
for p in pl:
    sid=f"s{p['n']:02d}"
    a=rgb(pathlib.Path('segments2/input')/sid/'seg.mp4')
    b=rgb(pathlib.Path('segments2/output')/sid/'synced.mp4')
    n=min(len(a),len(b)); a,b=a[:n],b[:n]
    la,lb=a.mean(),b.mean()
    dr,dg,db=[(b[...,i].mean()-a[...,i].mean()) for i in range(3)]
    print(f"{sid:6}{p['spk']:8}{la:9.2f}{lb:10.2f}{lb-la:+8.2f}{dr:+7.2f}{dg:+7.2f}{db:+7.2f}{b.std()/a.std():11.3f}")
