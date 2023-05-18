import matplotlib.pyplot as plt

class FootCycle:

    def __init__(self, dt, time,  phases = [(0.28, 0.0), (0.28, 0.075), (0.25, -0.0), (0.28, -0.075)], init_pos=[0,0]):
        self.phase = 0
        self.time = time
        self.dt = dt
        self.phss = phases
        self.current_t = 0
        self.init_phase = 0
        temp = init_pos
        self.initial_position = [init_pos[0], init_pos[1]]
        self.initialized = False
        self.pos = init_pos
        self.counter = 0

    def next_pos(self):

        self.pos[0] += len(self.phss)/self.time*self.dt*(self.phss[self.phase][0] - self.phss[self.phase - 1][0])
        self.pos[1] += len(self.phss)/self.time*self.dt*(self.phss[self.phase][1] - self.phss[self.phase - 1][1])
        if self.current_t > self.time/len(self.phss):
            self.phase += 1
            self.phase = self.phase % len(self.phss)
            self.current_t = 0
        self.current_t += self.dt

    def set_initial_phase(self, phase):
        self.init_phase = phase

    def initial_phase(self):
        self.pos[0] += len(self.phss)/self.time*self.dt*(self.phss[self.init_phase][0] - self.initial_position[0])
        self.pos[1] += len(self.phss)/self.time*self.dt*(self.phss[self.init_phase][1] - self.initial_position[1])
        self.current_t += self.dt
        self.counter += 1
        if self.current_t > self.time/len(self.phss):
            self.phase = self.init_phase + 1
            self.phase = self.phase % len(self.phss)
            self.current_t = 0
            return True
        return False
    
    def run(self):
        if self.initialized:
            self.next_pos()
        else:
            done = self.initial_phase()
            self.initialized = done


if __name__ == "__main__":
    initial_pos = [.25, -0.0]
    foot1 = FootCycle(1/180, 0.75, init_pos=initial_pos)
    foot1.set_initial_phase(2)
    x = []
    y = []
    t = []
    time = 0
    for i in range(int(5/foot1.dt)):
        if ((foot1.pos[0])**2 + (foot1.pos[1])**2)**(0.5) > 0.30004:
            print('bad')

        x.append(foot1.pos[0])
        y.append(foot1.pos[1])
        t.append(time)
        time += foot1.dt
        foot1.run()
    plt.figure(0)
    plt.plot(t, x, label='Y Pos.')
    plt.plot(t, y, label='X Pos.')
    plt.title('Foot Path of Walking Gait')
    plt.xlabel('Time (sec)', )
    plt.ylabel('Position (m)')
    plt.legend()

    plt.figure(1)
    plt.plot(y, x)
    plt.title('Walking Gait Pattern')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()
