import numpy as np, sys
sys.path.insert(0,'.')
from hand import link_pos
m=np.loadtxt('b.csv',delimiter=','); out=m.copy(); fps=30; C=slice(7+15,7+22)
s0,t0,t1,end=220,324,356,385   # 7.33 s -> 10.80 s, until 11.87 s (src 8.40), default at 12.83 s
off=t0-s0; X=int(sys.argv[1]) if len(sys.argv)>1 else 6   # crossfade half-width in frames
for i in range(t0-X,t1+1):
    cp=m[i-off,C]
    if i<t0+X:
        s=(i-(t0-X))/(2*X); w=s*s*(3-2*s); out[i,C]=(1-w)*m[i,C]+w*cp
    else: out[i,C]=cp
p0=out[t1,C]; v0=float(sys.argv[2])*(m[t1-off+1,C]-m[t1-off-1,C])*fps/2; p1=m[end,C]; T=(end-t1)/fps
for i in range(t1+1,end):
    s=(i-t1)/(end-t1)
    out[i,C]=(1-10*s**3+15*s**4-6*s**5)*p0+(s-6*s**3+8*s**4-3*s**5)*T*v0+(10*s**3-15*s**4+6*s**5)*p1
np.savetxt('d.csv',out,delimiter=',',fmt='%.6f')
q=out[:,7:]; v=np.diff(q[:,15:22],axis=0)*fps; a=np.diff(v,axis=0)*fps
src_v=np.diff(m[:,C],axis=0)*fps; src_a=np.diff(src_v,axis=0)*fps
print('seam1 (10.9-11.4) max vel',np.abs(v[327:342]).max().round(1),'max acc',np.abs(a[327:342]).max().round(0))
print('copy part max acc',np.abs(a[342:372]).max().round(0),' end blend max vel',np.abs(v[372:385]).max().round(1),'acc',np.abs(a[371:385]).max().round(0))
print('orig dance arm acc p99',np.percentile(np.abs(src_a[:330]),99).round(0),'max',np.abs(src_a[:330]).max().round(0))
for i in range(320,388,4): print(f"{i/30:5.2f} sh_p {q[i,15]:6.2f} hand {link_pos(q[i],'left_wrist_yaw_link').round(2)}")
