# Reinforcement Learning: explicit demo

## What it is all about

Consider the simplest static version of the "frozen lake" game:

<p align="center">
  <img src="png/000000/0000.png" width=50% />
</p>

where the red squares represent the ice traps: if the mover (small blue square) hits one of them the gave is over with the reward "-1" (lose). The green squares are exits: hiting one of them, the game is over with the reward "+1" (win). 

After being randomly generated the environment is fixed. Traing goes over episodes. On each episode the mover appears in free random field and does the steps until he pops into trap or successfully exits.