import subprocess, numpy as np, cv2, json, pathlib, sys
W,H=540,960
DET=cv2.FaceDetectorYN.create('models/yunet.onnx','',(W,H),0.6,0.3,5000)
def rd(p):
    r=subprocess.run(["ffmpeg","-v","error","-i",str(p),"-fps_mode","passthrough",
        "-vf",f"scale={W}:{H}","-pix_fmt","bgr24","-f","rawvideo","-"],capture_output=True).stdout
    return np.frombuffer(r,np.uint8).reshape(-1,H,W,3).astype(np.float32)
print(f"{'seg':6}{'face dLuma':>12}{'NONface dLuma':>15}{'uniform?':>10}")
for sid in sys.argv[1:]:
    a=rd(f'segments2/input/{sid}/seg.mp4'); b=rd(f'segments2/output/{sid}/synced.mp4')
    n=min(len(a),len(b)); a,b=a[:n],b[:n]
    fm=np.zeros((n,H,W),bool)
    for i in range(n):
        _,fs=DET.detect(np.ascontiguousarray(a[i].astype(np.uint8)))
        if fs is not None:
            for f in fs:
                x,y,w,h=[int(v) for v in f[:4]]
                fm[i,max(0,y):y+h,max(0,x):x+w]=True
    fa=a[fm].mean() if fm.any() else float('nan'); fb=b[fm].mean() if fm.any() else float('nan')
    na=a[~fm].mean(); nb=b[~fm].mean()
    df,dn=fb-fa, nb-na
    print(f"{sid:6}{df:+12.2f}{dn:+15.2f}{'YES' if abs(df-dn)<0.8 else 'NO':>10}")
