#  common definitions

class Waypoint(object):
    def __init__(self, x, y, s, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.s = s
        # TODO: add more properties

class WaypointLocal(Waypoint):
    def __init__(self, x, y, yaw, s, z=0):
        super(WaypointLocal, self).__init__(x, y, s, z)
        self.yaw = yaw
        # TODO: add more properties


class Egostate(object):
    def __init__(self, x=0, y=0, z=0, yaw=0, vel=0, acc=0, ts=0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.vel = vel
        self.acc = acc
        self.ts = ts