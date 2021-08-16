import numpy as np
#xpos is the x position at which to measure the location of the ball whenever it reaches there
#later will return the time it reaches there, right now don't have to worry about that

BALL_MASS = 2.7 # in grams
VT = 8 # terminal velocity of ping pong ball in meters/s
G = 9.8 # in meters/second
x_unit = np.array([1, 0, 0])
y_unit = np.array([0, 1, 0])

# ball_precise_location_world_frame = (x, y, z) in meters
def calculate_trajectory(ball_precise_location_world_frame, previous_ball_precise_location_world_frame, delta_time, x):

    # first, find xpos distance in rotated plane:
        # find projection of velocity vector on x-y plane
        # find point (x, y) where projection of velocity vector with tail at (x,y) of ball_precise_location_world_frame
        # intersects line x = xpos in 2d
        # the new xpos is sqrt(x^2 + y^2)

    # initial velocity
    velocity_vector = (ball_precise_location_world_frame - previous_ball_precise_location_world_frame)/delta_time

    # velocity vector projected on x-y plane
    proj_vel_vector = (np.dot(velocity_vector, x_unit) * x_unit + np.dot(velocity_vector, y_unit) * y_unit)[0:2]

    # line in direction of proj_vel_vector through the (x,y) of ball_precise_location_world_frame
    # parametrization of line:
    # x = ball_precise_location_world_frame[0] + proj_vel_vector[0]*s
    # y = ball_precise_location_world_frame[1] + proj_vel_vector[1]*s

    # plug in x and solve for s:
    s = (x - ball_precise_location_world_frame[0]) / proj_vel_vector[0]

    # plug that s into y to find the y location of that point:
    y = ball_precise_location_world_frame[1] + proj_vel_vector[1]*s

    # the point where the ball will reach the specified x position is (x, y).
    # the x position in the rotated plane is sqrt(x^2 + y^2).
    rotated_xpos = np.sqrt(x**2 + y**2)

    # then, use projectile motion in 2d plane to find z location at x = rotated_xpos:
    # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html

    # initial speed
    v0 = np.linalg.norm(velocity_vector)

    # launch angle
    theta = np.arctan2(velocity_vector[2], np.sqrt(velocity_vector[0]**2 + velocity_vector[1]**2))

    z = (VT/G)*(v0*np.sin(theta) + VT + VT*np.log(1 - rotated_xpos*G/(v0*VT*np.cos(theta))))

    print(x, y, z)

if __name__ == "__main__":
    calculate_trajectory(np.array([1, 2, 1]), np.array([0, 0, 0]), 0.1, 3)
