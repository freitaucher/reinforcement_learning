# Reinforcement Learning: basic demo

## What it is all about

Consider the simplest static version of the "frozen lake" game:

<p align="center">
  <img src="png/000000/0000.png" width=50% />
</p>

where the red squares represent the "ice" traps: if the mover (small blue square) hits one of them the game is over with the reward "-1" (lose). The green squares are "exits": by hitting one of them, the game is over with the reward "+1" (win). 

After being randomly generated the environment is fixed (saved as numpy array in `env.npz`) and the agent is trained to learn it. Training goes over episodes. In each episode the mover is randomly placed in a free field and steps until pops into trap or successfully exits. In the training regime, steps are done either according to the q-table or completely randomly. This is regulated by randomly generated $0<\epsilon<1$: if $\epsilon>0.5$, the step is selected as suggested by the qtable, or taken randomly, if otherwise. Random steps are needed in order to learn the environment; it also guarantees that the training episode will be finite, e.g. if the mover is getting arrested in a particular environment landscape.

With the `step` taken from `s` position (`s+step -> s_new`) the corresponding entry in the q-table is getting updated (and saved as numpy array in `qtable.npy`):
```
qtable[s,step] = q_table  + lr * (reward[s_new] + gamma*qtable[s_new, step_new] - qtable), 
```
where `0<lr<1` and `0<gamma<1`  are the fixed learning rate and discount parameters.  The `step_new` is the new step from the new position `s_new` suggested by the q-table (it might not be necessarily taken).

The training episodes are repeated until the q-table is getting stable, i.e. does not change anymore. Once it is stabilized one can run the test episodes, just by setting the $\epsilon=1$  which guarantees that all steps are taken according to the qtable.



##