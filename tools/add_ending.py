"""Usage: python add_ending.py in_30hz.csv out.csv T_CUT T_BLEND T_HOLD   (e.g. 11.40 1.5 1.5)

Append a calm ending to a 30 Hz G1 CSV: cut at a slow frame, blend (quintic, velocity-matched) to the
Unitree default standing pose while the stance foot stays pinned, then hold."""
import sys, numpy as np
sys.path.insert(0,'/mnt/project-files/g1_pilot/tools')
from motion_check import SDK, load_urdf, foot_sole, quat_xyzw, load_motion, axis_angle
import os
U=os.path.expanduser(os.environ.get("G1_URDF", "~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"))
J,L=load_urdf(U)
src,dst,t_cut,T_blend,T_hold=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])
fps,rp,rq,q=load_motion(src,30)
k=int(round(t_cut*fps))
DEF={"left_hip_pitch_joint":-0.312,"right_hip_pitch_joint":-0.312,"left_knee_joint":0.669,"right_knee_joint":0.669,
     "left_ankle_pitch_joint":-0.363,"right_ankle_pitch_joint":-0.363,"left_elbow_joint":0.6,"right_elbow_joint":0.6,
     "left_shoulder_roll_joint":0.2,"left_shoulder_pitch_joint":0.2,"right_shoulder_roll_joint":-0.2,"right_shoulder_pitch_joint":0.2}
qT=np.array([DEF.get(n,0.0) for n in SDK])
v0=(q[k+1]-q[k-1])*fps/2
# quintic: p(0)=q0, p'(0)=v0, p''(0)=0, p(T)=qT, p'(T)=0, p''(T)=0
T=T_blend; nb=int(round(T*fps)); nh=int(round(T_hold*fps))
def quintic(t):
    s=t/T; d=qT-q[k]
    h=lambda a,b,c: a*s**3+b*s**4+c*s**5
    # position part (rest-to-rest) + velocity part
    pos=q[k]+d*(10*s**3-15*s**4+6*s**5)
    vel=v0*T*(s-6*s**3+8*s**4-3*s**5)
    return pos+vel
qs=[quintic((i+1)/fps) for i in range(nb)]+[qT]*nh
# root orientation: from cut quat to yaw-only of cut, slerp with same smoothstep
def q2m(x): return quat_xyzw(*x)
x,y,z,w=rq[k]; yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
qyaw=np.array([0,0,np.sin(yaw/2),np.cos(yaw/2)])
def slerp(a,b,s):
    d=np.dot(a,b)
    if d<0: b=-b; d=-d
    if d>0.9995: r=a+s*(b-a); return r/np.linalg.norm(r)
    th=np.arccos(d); return (np.sin((1-s)*th)*a+np.sin(s*th)*b)/np.sin(th)
fz0=[foot_sole(J,rp[k],q2m(rq[k]),dict(zip(SDK,q[k])),s) for s in ('left','right')]
stance='left' if fz0[0][:,2].min()<fz0[1][:,2].min() else 'right'
anchor=foot_sole(J,rp[k],q2m(rq[k]),dict(zip(SDK,q[k])),stance).mean(0); anchor[2]=min(anchor[2], anchor[2])
rows=[]; minswing=9; lastfeet=None
for i,qi in enumerate(qs):
    s=min(1.0,(i+1)/nb); ss=10*s**3-15*s**4+6*s**5
    Rq=slerp(rq[k],qyaw,ss); R=q2m(Rq)
    rel=foot_sole(J,np.zeros(3),R,dict(zip(SDK,qi)),stance).mean(0)
    root=anchor-rel
    # lift/lower so the lowest sole point of the stance foot is at the cut's height, ending at 0
    fs=foot_sole(J,root,R,dict(zip(SDK,qi)),stance); root[2]+= (fz0[0 if stance=='left' else 1][:,2].min()*(1-ss)) - fs[:,2].min()
    sw=foot_sole(J,root,R,dict(zip(SDK,qi)),'right' if stance=='left' else 'left')
    minswing=min(minswing,sw[:,2].min()); lastfeet=(fs[:,2].min(),sw[:,2].min(),sw[:,:2].mean(0),fs[:,:2].mean(0))
    rows.append(np.r_[root,Rq,qi])
out=np.r_[np.c_[rp[:k+1],rq[:k+1],q[:k+1]],np.array(rows)]
np.savetxt(dst,out,delimiter=',',fmt='%.6f')
qq=out[:,7:]; vel=np.diff(qq,axis=0)*fps; acc=np.diff(vel,axis=0)*fps
lo=np.array([L[n][0] for n in SDK]); hi=np.array([L[n][1] for n in SDK])
print('stance',stance,'cut frame',k,'v0 max',np.abs(v0).max().round(2))
print('frames',len(out),'dur',len(out)/fps)
print('swing foot min z during blend cm',round(minswing*100,2),'end feet z cm',np.round(np.array(lastfeet[:2])*100,2),'stance width m',round(float(np.linalg.norm(lastfeet[2]-lastfeet[3])),3))
print('max |vel| blend',np.abs(vel[k:]).max().round(2),'orig',np.abs(vel[:k]).max().round(2))
print('max |acc| blend',np.abs(acc[k-1:]).max().round(1),'orig p99',np.percentile(np.abs(acc[:k-1]),99).round(1))
print('limit viol blend frames',int(((qq[k:]<lo)|(qq[k:]>hi)).any(1).sum()))
print('root z end',out[-1,2].round(3))
