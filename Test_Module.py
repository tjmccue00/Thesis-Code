import QLearning as ql
import numpy as np



if __name__ == "__main__":

   qlearn = ql.qlearning(10, 4, [-200,200], [(-.62,.62),(-1.57,1.57),(-0.5,0.5),(-0.5,-0.5)], 0.95, 0.1, 30)

   bins = np.linspace(-.5,.5,10)
   x = 0.13
   print(np.digitize(x, bins))