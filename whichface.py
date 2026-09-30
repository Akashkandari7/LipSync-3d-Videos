import subprocess,numpy as np,cv2,sys,json
W,H=540,960
DET=cv2.FaceDetectorYN.create('models/yunet.onnx','',(W,H),0.6,0.3,5000)
def rd(p):
    r=subprocess.run(['ffmpeg','-v','error','-i',p,'-fps_mode','passthrough','-vf',f'scale={W}:{H}',
        '-pix_fmt','bgr24','-f','rawvideo','-'],capture_output=True).stdout
    return np.frombuffer(r,np.uint8).reshape(-1,H,W,3).astype(np.int16)
sid=sys.argv[1]
i=rd(f'segments2/input/{sid}/seg.mp4')
o=rd(f'segments2/output/{sid}/synced_OLD.mp4')
n=rd(f'segments2/output/{sid}/synced.mp4')
m=min(len(i),len(o),len(n))
# collect face mouth boxes from a few sample frames of the INPUT
boxes={}
for f in range(0,m,max(1,m//8)):
    _,fs=DET.detect(np.ascontiguousarray(i[f].astype(np.uint8)))
    if fs is None: continue
    for fa in fs:
        rx,ry,lx,ly=fa[10],fa[11],fa[12],fa[13]
        cx,cy=(rx+lx)/2,(ry+ly)/2; half=max(abs(lx-rx),8.)
        key=(round(cx/60),round(cy/60))
        boxes.setdefault(key,[]).append((int(cx-half),int(cy-half*.7),int(cx+half),int(cy+half*.7)))
print(f'{sid}: {len(boxes)} face(s) found')
print(f"{'face @(x,y)':>16}{'OLD changed':>14}{'NEW changed':>14}  verdict")
for key,bs in sorted(boxes.items(), key=lambda kv:-len(kv[1])):
    b=np.median(np.array(bs),axis=0).astype(int)
    x0,y0,x1,y1=max(0,b[0]),max(0,b[1]),min(W,b[2]),min(H,b[3])
    if x1<=x0 or y1<=y0: continue
    do=np.abs(i[:m,y0:y1,x0:x1]-o[:m,y0:y1,x0:x1]).mean()
    dn=np.abs(i[:m,y0:y1,x0:x1]-n[:m,y0:y1,x0:x1]).mean()
    v='NEW syncs this face' if dn>do*1.25 else ('OLD synced this one' if do>dn*1.25 else 'similar')
    print(f"{str((x0,y0)):>16}{do:14.2f}{dn:14.2f}  {v}")
