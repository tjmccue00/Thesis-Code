import Environments.AssetClass.Foot_Cycle as fc
import matplotlib.pyplot as plt

class Gait():

    def __init__(self, dt, time, phases, initial_phases, foot_poss):

        self.dt = dt
        self.phases = phases
        self.initial_phases = initial_phases
        self.initial_pos = foot_poss
        self.time = time

        self.front_right = fc.FootCycle(self.dt, time, phases=self.phases, init_pos=self.initial_pos[0])
        self.front_left = fc.FootCycle(self.dt, time, phases=self.phases, init_pos=self.initial_pos[1])
        self.back_right = fc.FootCycle(self.dt, time, phases=self.phases, init_pos=self.initial_pos[2])
        self.back_left = fc.FootCycle(self.dt, time, phases=self.phases, init_pos=self.initial_pos[3])

        self.front_right.set_initial_phase(self.initial_phases[0])
        self.front_left.set_initial_phase(self.initial_phases[1])
        self.back_right.set_initial_phase(self.initial_phases[2])
        self.back_left.set_initial_phase(self.initial_phases[3])



    def run(self):
        self.front_right.run()
        self.front_left.run()
        self.back_right.run()
        self.back_left.run()

        fr_pos = (self.front_right.pos[0], self.front_right.pos[1])
        fl_pos = (self.front_left.pos[0], self.front_left.pos[1])
        br_pos = (self.back_right.pos[0], self.back_right.pos[1])
        bl_pos = (self.back_left.pos[0], self.back_left.pos[1])

        return [fr_pos, fl_pos, br_pos, bl_pos]
    
if __name__ == "__main__":
    xy = [0.0, 0.3004]
    init_feet_pos = [xy, [0,0], [0,0], [0,0]]
    phase = [(0.28, 0.0), (0.28, 0.05), (0.25, -0.0), (0.28, -0.05)] #3-0.25
    walk = Gait(1/180, 5, phase, [0, 1, 2, 3], init_feet_pos)
    time = 0
    x = []
    y = []
    t = []
    for i in range(180*10):
        feet_pos = walk.run()
        #print(feet_pos[1][0], feet_pos[1][1])
        time += 1/180
        x.append(feet_pos[0][0])
        y.append(feet_pos[0][1])
        t.append(time)

    plt.plot(t, x)
    plt.plot(t, y)
    plt.show()