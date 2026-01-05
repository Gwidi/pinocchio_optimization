import pinocchio
import numpy as np
from numpy.linalg import norm, solve

class Robot:
    def __init__(self, urdf_path):
        # Load the urdf model
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        # Create data required by the algorithms
        self.data = self.model.createData()

    def compute_forward_kinematics(self, q):
        # Perform the forward kinematics over the kinematic tree
        pinocchio.forwardKinematics(self.model, self.data, q)
        # Update all the frame placements
        pinocchio.updateFramePlacements(self.model, self.data)
        # Print the position of the last frame (end-effector)
        print("End-effector position:", self.data.oMf[-1].translation)

    def compute_inverse_kinematics(self, oMdes):
        
        JOINT_ID = 7
        q = pinocchio.neutral(self.model)
        eps    = 1e-4
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-12
        
        i=0
        while True:
            pinocchio.forwardKinematics(self.model,self.data,q)
            dMi = oMdes.actInv(self.data.oMi[JOINT_ID])
            err = pinocchio.log(dMi).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            J = pinocchio.computeJointJacobian(self.model,self.data,q,JOINT_ID)
            v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.model,q,v*DT)
            # if not i % 10:
            #     print('%d: error = %s' % (i, err.T))
            i += 1
        
        if success:
            print("Convergence achieved!")
        else:
            print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
        
        print('\nresult: %s' % q.flatten().tolist())
        print('\nfinal error: %s' % err.T)

    def compute_jacobian(self, q, frame_id):
        # Compute the Jacobian at the specified frame
        pinocchio.computeJointJacobians(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        J = pinocchio.getFrameJacobian(self.model, self.data, frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
        return J
    
    def compute_forward_dynamics(self, q, dq):
        # Compute the joint accelerations given the current state and torques
        pinocchio.aba(self.model, self.data, q, dq, np.zeros(self.model.nv))
        print("Joint accelerations:", self.data.ddq)
    
    def compute_inverse_dynamics(self, q, dq, ddq):
        # Compute the required joint torques for given accelerations
        pinocchio.rnea(self.model, self.data, q, dq, ddq)
        print("Required joint torques:", self.data.tau)