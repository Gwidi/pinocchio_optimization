import pinocchio

class Robot:
    def __init__(self, urdf_path="./iiwa_cup.urdf"):
        # Load the urdf model
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        # Create data required by the algorithms
        self.data = self.model.createData()
        # Sample random joint configuration
        self.q = pinocchio.neutral(self.model)

    def set_joint_positions(self, joint_positions):
        if len(joint_positions) != self.model.nq:
            raise ValueError("Joint positions length does not match number of joints.")
        self.q = joint_positions

    def compute_forward_kinematics(self):
        # Perform the forward kinematics over the kinematic tree
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        # Update all the frame placements
        pinocchio.updateFramePlacements(self.model, self.data)
        # Print the position of the last frame (end-effector)
        print("End-effector position:", self.data.oMf[-1].translation)

    def get_end_effector_position(self, frame_name):
        frame_id = self.model.getFrameId(frame_name)
        if frame_id == -1:
            raise ValueError(f"Frame '{frame_name}' not found in the model.")
        return self.data.oMf[frame_id].translation