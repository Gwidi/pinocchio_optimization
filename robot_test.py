from robot import Robot
from environment import Environment
import pinocchio as pino
import numpy as np

def main():
    urdf_path = "iiwa_cup.urdf"
    robot = Robot(urdf_path=urdf_path)
    q = np.array([1., 1., 1., 1., 1., 1., 1.])
    # Compute the forward kinematics
    robot.compute_forward_kinematics(q=q)
    # Print all frame names
    for i, frame in enumerate(robot.model.frames):
        print(f"Frame {i}: {frame.name}")
    # Compute the Jacobian at the end-effector frame
    J = robot.compute_jacobian(q=q, frame_id=robot.model.getFrameId("F_link_ee"))
    print("Jacobian at end-effector frame:\n", J)

    robot.compute_forward_dynamics(q=q, dq=np.ones(robot.model.nv))
    robot.compute_inverse_dynamics(q=q, dq=np.ones(robot.model.nv), ddq=np.ones(robot.model.nv))

    # Test inverse kinematics
    robot.compute_inverse_kinematics(oMdes=pino.SE3(np.eye(3), np.array([-8.52966496e-17,  1.09245946e-16,  1.26100000e+00])))
    robot.compute_inverse_kinematics(oMdes=pino.SE3(np.eye(3), np.array([0.46844282, 0.23633956, 0.93455383])))



if __name__ == "__main__":
    main()