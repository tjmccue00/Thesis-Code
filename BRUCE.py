import math

class Bruce(object):

    def __init__(self, hip_len, knee_len, hip_hip_len, joint_offset):

        self.lengths = {"Knee": knee_len, "Hip": hip_len, "Hip2Hip": hip_hip_len}
        self.joint_cc_set = joint_offset
        self.stand = [(0,0), (0,0), (0,0), (0,0)]


    def set_stand(self, fr, fl, br, bl):
        self.stand = [fr, fl, br, bl]

    def stand(self):
        

        return 

    def inverse_kin(self, x,y):

        phi_knee = -math.acos((x**2 + y**2 - self.lengths["Knee"]**2 - self.lengths["Hip"]**2)/
                               (2*self.lengths["Hip"]*self.lengths["Knee"]))

        phi_hip = math.atan2(-y, x) - math.atan2((self.lengths["Knee"]*math.sin(phi_knee)),
                               (self.lengths["Hip"] + self.lengths["Knee"]*math.cos(phi_knee)))

        return [phi_knee, phi_hip]

    def kin(self, phi_knee, phi_hip):
        x = self.lengths["Hip"]*math.sin(phi_hip) + self.lengths["Knee"]*math.sin(phi_hip + phi_knee)
        y = self.lengths["Hip"]*math.cos(phi_hip) + self.lengths["Knee"]*math.cos(phi_hip + phi_knee)

        return [x,y]

    def normalize_joints(self, joint):
        while joint > math.pi:
            joint -= math.pi
        while joint < -math.pi:
            joint += math.pi
        return joint
    
if __name__ == "__main__":
     
    brucie = Bruce(0.14, 0.16, 0.2, [1,1,1,1])

    x, y = brucie.inverse_kin(-0.10, 0.20)

    print(brucie.kin(x, y))