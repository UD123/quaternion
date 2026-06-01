"""
quaternion_plot.py - This file defines the core Quaternion class

             /\
   o========/==\

Parameters:
- centerPoint: The starting point in 3D space.
- rotationVector: The rotation vector for the flag direction.
- flagSize: The size of the flag (length).
- name: The name of the flag.

"""

import numpy as np
from quaternion import Quaternion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class QuaternionPlot(Quaternion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        # for plot - graphics handle
        self.qh         = None

    def rotation_matrix_from_axis_angle(self):
        """
        Calculates the 3D rotation matrix for a given axis and angle 
        using Rodrigues' rotation formula.

        Args:
            axis (np.ndarray): The 3D vector representing the rotation axis. 
                            It will be normalized inside the function.
            angle_rad (float): The rotation angle in radians.

        Returns:
            np.ndarray: The 3x3 rotation matrix.
        """
        axis, angle_rad = self.vector, self.angle

        # 1. Normalize the rotation axis (k)
        k = axis / np.linalg.norm(axis)
        kx, ky, kz = k
        
        # 2. Pre-calculate sine, cosine, and (1 - cosine)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        one_minus_c = 1.0 - c

        # 3. Define the skew-symmetric cross-product matrix (K)
        K = np.array([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0]
        ])

        # 4. Apply Rodrigues' Rotation Formula (R = I + s*K + (1-c)*K^2)
        
        # Option 1: Direct formula (as written above)
        # R = np.identity(3) + s * K + one_minus_c * (K @ K)
        
        # Option 2: Using the expanded form (often more stable/faster)
        R = np.array([
            [c + kx*kx*one_minus_c, kx*ky*one_minus_c - kz*s, kx*kz*one_minus_c + ky*s],
            [ky*kx*one_minus_c + kz*s, c + ky*ky*one_minus_c, ky*kz*one_minus_c - kx*s],
            [kz*kx*one_minus_c - ky*s, kz*ky*one_minus_c + kx*s, c + kz*kz*one_minus_c]
        ])
        
        return R     

    def quaternion_to_rotation_matrix(self):
        """
        Converts a unit quaternion (w, x, y, z) to a 3x3 rotation matrix.
        
        Args:
            q (np.array): A 4-element NumPy array [w, x, y, z] (scalar-first convention).
            
        Returns:
            np.array: A 3x3 rotation matrix.
        """
        w, x, y, z = self.q
        
        # Ensure the quaternion is a unit quaternion for a valid rotation
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            return np.identity(3)
        w, x, y, z = self.q / norm
        
        # Rotation Matrix (R) calculation
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return R       

    # rotate vectors 
    def rotation_matrix_from_vectors(self, v1, v2):
        """
        Calculates the 3D rotation matrix that transforms vector v1 into vector v2.

        Args:
            v1 (np.ndarray): The starting 3D vector.
            v2 (np.ndarray): The ending 3D vector.

        Returns:
            np.ndarray: The 3x3 rotation matrix R such that R @ v1 is parallel to v2.
        """
        # 1. Normalize the vectors
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)

        # Calculate cos(theta) and sin(theta)
        cos_theta = np.dot(u1, u2)

        # Edge Case 1: Vectors are the same (or parallel, angle = 0)
        if np.isclose(cos_theta, 1.0):
            return np.identity(3)

        # Edge Case 2: Vectors are opposite (anti-parallel, angle = 180 degrees)
        if np.isclose(cos_theta, -1.0):
            # Find an arbitrary rotation axis k that is perpendicular to u1.
            # This is a bit arbitrary, but a simple way is to use a cross-product
            # with the vector that is least parallel to u1.
            
            # Determine the least parallel standard basis vector for the cross product
            # to ensure the resulting vector is non-zero.
            k_candidate = np.array([0., 0., 0.])
            abs_u1 = np.abs(u1)
            
            if abs_u1[0] < abs_u1[1] and abs_u1[0] < abs_u1[2]:
                k_candidate[0] = 1.0
            elif abs_u1[1] < abs_u1[0] and abs_u1[1] < abs_u1[2]:
                k_candidate[1] = 1.0
            else:
                k_candidate[2] = 1.0
                
            k = np.cross(u1, k_candidate)
            k = k / np.linalg.norm(k) # Normalize the axis
            
            # For 180-degree rotation (cos_theta=-1), the formula simplifies:
            # R = I + 2 * (k * k.T - I) = 2 * k * k.T - I
            # Or, using the full formula with c=-1, s=0:
            # R_ij = k_i * k_j * (1 - (-1)) - delta_ij * (-1)
            # R_ij = 2 * k_i * k_j + delta_ij
            
            # Calculate the outer product matrix K_outer = k * k.T
            K_outer = np.outer(k, k)
            return 2 * K_outer - np.identity(3)

        # Standard Case

        # 2. Calculate the rotation axis (k)
        v = np.cross(u1, u2)
        s = np.linalg.norm(v) # sin(theta) is the length of the cross product
        k = v / s # Normalized rotation axis

        # 3. Define the skew-symmetric cross-product matrix (K)
        kx, ky, kz = k
        K = np.array([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0]
        ])

        # 4. Apply Rodrigues' Rotation Formula
        # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        
        I = np.identity(3)
        c = cos_theta
        
        # K^2 = K @ K
        K_sq = K @ K 
        R = I + s * K + (1 - c) * K_sq
        
        return R

    # -----------------------------
    # visualization
    def plot(self, ax = None, clr = 'b'):
        """ Plots quaternion flag on the axis specified.

        Updates:
            qh - handle of the flag which is already been rendered
        Inputs:
            ax - axis if already created
           clr - colors 'r','b',...
        Returns:
            ax - handle to the axis
        """
        # flag size is proportinal to the quarternion norm
        height          = self.norm
        flag_height     = height/4

        # flag shape
        xyz_flag        = np.ones((4,5))
        xyz_flag[0:3,0] = [0, 0, 0]
        xyz_flag[0:3,1] = [height, 0, 0]
        xyz_flag[0:3,2] = [height - flag_height/2,  0, flag_height/2]
        xyz_flag[0:3,3] = [height - flag_height,0, 0]
        xyz_flag[0:3,4] = [0, 0, 0]

        # transform - not good
        #xyz_tranformed = np.dot(self.transformation_matrix,xyz_flag)

        # # another option
        xyz_tranformed = xyz_flag.copy()
        for k in range(xyz_flag.shape[1]):
            xyz_tranformed[:3,k] = self.rotate(xyz_flag[:3,k])
            #xyz_tranformed[:3,k] = self.rotate_without_normalization(xyz_flag[:3,k])

        # using transformation matrx from 2 vecors
        #R = self.rotation_matrix_from_vectors(self.vector, xyz_flag[0:3,1])
        #R = self.rotation_matrix_from_axis_angle()
        #R = self.quaternion_to_rotation_matrix()

        # xyz_tranformed = xyz_flag.copy()
        # for k in range(xyz_flag.shape[1]):
        #     xyz_tranformed[:3,k] = np.dot(R,xyz_flag[:3,k])            

        # render
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
            ax.set_aspect('equal')
            ax.set_xlim(-height,height)
            ax.set_ylim(-height,height)
            ax.set_zlim(-height,height)
            ax.set_xlabel('i')
            ax.set_ylabel('j')
            ax.set_zlabel('k')             

        if self.qh is None:
            self.qh,  = ax.plot3D(xyz_tranformed[0,:], xyz_tranformed[1,:], xyz_tranformed[2,:], color=clr)
        else:
            self.qh.set_data(xyz_tranformed[0,:], xyz_tranformed[1,:])
            self.qh.set_3d_properties(xyz_tranformed[2,:])             

        return ax

    def delete(self):
        "deletes the Box and frame from the plot"
        if self.qh is not None:
            self.qh.remove()


#%% ---------------------------------------
import unittest
class TestQuaternionPlot(unittest.TestCase):

    def test_quaternion_plot(self):
        q1 = QuaternionPlot(axis=[0, 1, 0], degrees=90) # Create another quaternion representing no rotation at all
        print(q1)
        ax = q1.plot(clr = 'r')
        plt.show()

        self.assertIsInstance(q1, QuaternionPlot)

    def test_quaternion_multi_plot(self):
        q1 = QuaternionPlot(axis=[1, 0, 0], degrees=90) # Create another quaternion representing no rotation at all
        q2 = QuaternionPlot(axis=[0, 1, 0], degrees=90) # Create a quaternion representing a rotation of +90 degrees about positive x axis.
        q3 = QuaternionPlot(axis=[0, 0, 1], degrees=90) # Create a quaternion representing a rotation of +180 degrees about positive y axis.
        q4 = QuaternionPlot(axis=[1, 1, 1], degrees=90)
        ax = q1.plot(clr = 'r')
        ax = q2.plot(ax, clr = 'g')
        ax = q3.plot(ax, clr = 'b')
        ax = q4.plot(ax, clr = 'c')
        plt.show()
        self.assertIsInstance(q2, QuaternionPlot)

    def test_quaternion_norm(self):
        "works when normalization of axis angle is disabled"
        q1 = QuaternionPlot(axis=[1, 0, 0], degrees=90) # Create another quaternion representing no rotation at all
        q2 = QuaternionPlot(axis=[.5, 0, 0], degrees=90) # Create a quaternion representing a rotation of +90 degrees about positive x axis.
        q3 = QuaternionPlot(axis=[.1, 0, 0], degrees=90) # Create a quaternion representing a rotation of +180 degrees about positive y axis.
        q4 = QuaternionPlot(axis=[0, .5, 0], degrees=90)
        ax = q1.plot(clr = 'r')
        ax = q2.plot(ax, clr = 'g')
        ax = q3.plot(ax, clr = 'b')
        ax = q4.plot(ax, clr = 'c')
        plt.show()
        self.assertIsInstance(q2, QuaternionPlot)   

    def test_show_multiplication(self):
        "rotate q1 usaing qr"
        q1 = QuaternionPlot(axis=[1, 0, 0], degrees=90) # Create another quaternion representing no rotation at all
        q2 = QuaternionPlot(axis=[0, 1, 0], degrees=90) # Create a quaternion representing a rotation of +90 degrees about positive x axis.
        q3 = q2 * q1
        q4 = q2.inverse
        q5 = q3 * q4
        ax = q1.plot(    clr = 'r')
        ax = q2.plot(ax, clr = 'g')
        ax = q3.plot(ax, clr = 'b')
        ax = q4.plot(ax, clr = 'c')
        ax = q5.plot(ax, clr = 'm')
        plt.show()
        self.assertIsInstance(q2, QuaternionPlot)             

if __name__ == '__main__':

    #unittest.main()  
    tst = TestQuaternionPlot()  
    #tst.test_quaternion_multi_plot() # ok
    #tst.test_quaternion_norm() # nok
    tst.test_show_multiplication()