import os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

base = 'experiments/drl_curriculum_20260529_180902'
phases = [('phase1_imitation','Phase1 Imitation'),('phase2_mixed','Phase2 Mixed'),('phase3_task_only','Phase3 Autonomous')]

fig,ax=plt.subplots(figsize=(14,5))
offset=0
colors=['#2E86AB','#A23B72','#F18F01']
all_r=[]

for i,(d,label) in enumerate(phases):
    rewards=[]
    with open(os.path.join(base,d,'monitor.csv')) as f:
        for line in f:
            line=line.strip()
            if line.startswith('#') or line.startswith('r,'): continue
            parts=line.split(',')
            if parts and parts[0]: rewards.append(float(parts[0]))
    if not rewards: continue
    mean_r=np.mean(rewards); max_r=np.max(rewards)
    print(f'{label}: {len(rewards)} eps, mean={mean_r:.2f}, max={max_r:.2f}')
    all_r.extend(rewards)
    xs=list(range(offset,offset+len(rewards)))
    offset+=len(rewards)
    # Raw dots
    ax.plot(xs,rewards,alpha=0.15,lw=0.3,color=colors[i])
    # Rolling mean
    w=min(50,len(rewards)//5) if len(rewards)>10 else 1
    if w>1:
        roll=np.convolve(rewards,np.ones(w)/w,mode='valid')
        roll_xs=range(offset-len(rewards)+w-1,offset)
        ax.plot(roll_xs,roll,color=colors[i],lw=2,label=f'{label} mean={mean_r:.1f} max={max_r:.1f}')
    else:
        ax.plot(xs,rewards,color=colors[i],lw=2,label=f'{label} mean={mean_r:.1f} max={max_r:.1f}')

# Phase dividers
off=0
for d,_ in phases:
    with open(os.path.join(base,d,'monitor.csv')) as f:
        n=sum(1 for l in f if not l.startswith('#') and not l.startswith('r,') and l.strip())
    off+=n
    ax.axvline(off,color='gray',ls='--',alpha=0.3)

ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
ax.set_title('PPO Curriculum Training (20260529)')
ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=0.2)
out=os.path.join(base,'learning_curve_full.png')
fig.tight_layout(); fig.savefig(out,dpi=200,bbox_inches='tight')
print(f'[DONE] {out}')
print(f'Total: {len(all_r)} eps, global mean={np.mean(all_r):.1f}')
