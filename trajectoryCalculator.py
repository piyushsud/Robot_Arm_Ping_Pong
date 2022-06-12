import numpy as np
#xpos is the x position at which to measure the location of the ball whenever it reaches there
#later will return the time it reaches there, right now don't have to worry about that

BALL_MASS = 2.7 # in grams
VT = 8 # terminal velocity of ping pong ball in meters/s
G = 9.8 # in meters/second
x_unit = np.array([1, 0, 0])
y_unit = np.array([0, 1, 0])

class TrajectoryCalculator:

    def __init__(self):
        pass

    # ball_precise_location_world_frame = (x, y, z) in meters
    def calculate_trajectory(self, ball_precise_location_world_frame, previous_ball_precise_location_world_frame, delta_time, x):

        # convert frame because this code assumes positive x is towards the robot and positive y is away from the realsense
        ball_precise_location_world_frame[0] = -ball_precise_location_world_frame[0]
        ball_precise_location_world_frame[1] = -ball_precise_location_world_frame[1]
        previous_ball_precise_location_world_frame[0] = -previous_ball_precise_location_world_frame[0]
        previous_ball_precise_location_world_frame[1] = -previous_ball_precise_location_world_frame[1]
        x = -x


        # print(ball_precise_location_world_frame[0])

        # first, find xpos distance in rotated plane:
            # find projection of velocity vector on x-y plane
            # find point (x, y) where projection of velocity vector with tail at (x,y) of ball_precise_location_world_frame
            # intersects line x = xpos in 2d
            # the new xpos is sqrt(x^2 + y^2)

        # initial velocity
        velocity_vector = (ball_precise_location_world_frame - previous_ball_precise_location_world_frame)/delta_time

        # velocity vector projected on x-y plane
        proj_vel_vector = (np.dot(velocity_vector, x_unit) * x_unit + np.dot(velocity_vector, y_unit) * y_unit)[0:2]

        # line in direction of proj_vel_vector through the (x,y) of the ball
        # parametrization of line:
        # x = ball_precise_location_world_frame[0] + proj_vel_vector[0]*s
        # y = ball_precise_location_world_frame[1] + proj_vel_vector[1]*s

        ball_location = (ball_precise_location_world_frame + previous_ball_precise_location_world_frame)/2

        # plug in x and solve for s:
        s = (x - ball_location[0]) / proj_vel_vector[0]

        # plug that s into y to find the y location of that point:
        y = ball_location[1] + proj_vel_vector[1]*s

        # the point where the ball will reach the specified x position is (x, y).
        # the x position in the rotated plane is sqrt(x^2 + y^2).
        rotated_xpos = np.sqrt(x**2 + y**2)

        # then, use projectile motion in 2d plane to find z location at x = rotated_xpos:
        # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html

        # initial speed
        v0 = np.linalg.norm(velocity_vector)

        # launch angle
        theta = np.arctan2(velocity_vector[2], np.sqrt(velocity_vector[0]**2 + velocity_vector[1]**2))

        # upper_limit = v0*VT*np.cos(theta)/G
        upper_limit = 1000000
        if x < upper_limit:
            # np.log function is actually the log with an exponential base, not base 10
            # z = (VT/G)*((v0*np.sin(theta) + VT) * (G*rotated_xpos/(v0*VT*np.cos(theta))) + VT*np.log(1 - rotated_xpos*G/(v0*VT*np.cos(theta))))

            # t >> vt/g
            # z = (VT / G) * (v0 * np.sin(theta) + VT + VT * np.log(1 - rotated_xpos * G / (v0 * VT * np.cos(theta))))

            # t << vt/g (assuming air resistance is negligible)
            t = -(VT/G)*np.log(1 - G*rotated_xpos/(v0*VT*np.cos(theta)))
            z = v0*np.sin(theta)*t - (G/2)*t**2

            return np.array([-x, -y, z])
        else:
            print("ball cannot travel that far due to air resistance")
            return None


if __name__ == "__main__":
    trajectoryCalculator = TrajectoryCalculator()
    arr = trajectoryCalculator.calculate_trajectory(np.array([0.9, -0.2, 0.3]), np.array([1, -0.2, 0.2]), 0.2, 0.1)
    print(arr)
