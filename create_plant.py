import os
import numpy as np

import pydrake.common
from pydrake.all import (
    MultibodyPlant, Parser, ProcessModelDirectives, LoadModelDirectives,
    SpatialInertia, FindResourceOrThrow, RigidTransform, Joint,
)
from pydrake.math import RollPitchYaw

models_dir = os.path.join(
    os.path.dirname(os.curdir), 'models')

bubble_sdf_file = 'models/bubble_gripper/schunk_wsg_50_bubble_collision.sdf'


# L7 is link 7, E is end-effector
X_L7E = RigidTransform(
    RollPitchYaw(np.pi, 0, 0), np.array([0, 0, 0.114]))

def add_package_paths(parser: Parser):
    parser.package_map().Add(
        "drake_manipulation_models",
        os.path.join(pydrake.common.GetDrakePath(),
                     "manipulation/models"))

    parser.package_map().Add("iiwa_controller", models_dir)

def create_iiwa_soft_bubble_plant(gravity):
    plant = MultibodyPlant(time_step=2e-4)
    parser = Parser(plant=plant)
    add_package_paths(parser)
    ProcessModelDirectives(
        LoadModelDirectives(os.path.join(models_dir, 'iiwa.yml')),
        plant, parser)
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    gripper_plant = MultibodyPlant(2e-4)
    gripper_parser = Parser(gripper_plant)
    gripper_parser.AddModelFromFile(bubble_sdf_file)
    gripper_plant.Finalize()

    iiwa_model = plant.GetModelInstanceByName('iiwa')
    bubble_gripper = plant.AddRigidBody(
        'bubble', iiwa_model, calc_schunk_inertia())
    plant.WeldFrames(
        frame_on_parent_P=plant.GetFrameByName('iiwa_link_7', iiwa_model),
        frame_on_child_C=bubble_gripper.body_frame(),
        X_PC=X_L7E)

    plant.Finalize()
    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant.GetFrameByName("iiwa_link_" + str(i)).index())



    return plant, link_frame_indices


def calc_schunk_inertia():
    """
    Verbatim translation from a function in drake's ManipulationStation.
    :return:
    """
    def calc_finger_pose_in_gripper_frame(slider: Joint):
        # Pose of the joint's parent frame P (attached on gripper body G) in the
        #  frame of the gripper G.
        X_GP = slider.frame_on_parent().GetFixedPoseInBodyFrame()
        # Pose of the joint's child frame C (attached on the slider's finger
        #  body) in the frame of the slider's finger F.
        X_FC = slider.frame_on_child().GetFixedPoseInBodyFrame()
        # When the slider's translational dof is zero, then P coincides with C.
        # Therefore:
        X_GF = X_GP.multiply(X_FC.inverse())
        return X_GF

    def calc_finger_spatial_inertia_in_gripper_frame(
            M_FFo_F: SpatialInertia, X_GF: RigidTransform):
        """
        Helper to compute the spatial inertia of a finger F in about the
            gripper's origin Go, expressed in G.
        """
        M_FFo_G = M_FFo_F.ReExpress(X_GF.rotation())
        p_FoGo_G = -X_GF.translation()
        M_FGo_G = M_FFo_G.Shift(p_FoGo_G)
        return M_FGo_G

    plant = MultibodyPlant(1e-3)
    parser = Parser(plant)
    parser.AddModelFromFile(bubble_sdf_file)
    plant.Finalize()

    gripper_body = plant.GetBodyByName("body")
    left_finger_body = plant.GetBodyByName("left_finger")
    right_finger_body = plant.GetBodyByName("right_finger")

    M_GGo_G = gripper_body.default_spatial_inertia()
    M_LLo_L = left_finger_body.default_spatial_inertia()
    M_RRo_R = right_finger_body.default_spatial_inertia()
    left_slider = plant.GetJointByName("left_finger_sliding_joint")
    right_slider = plant.GetJointByName("right_finger_sliding_joint")

    X_GL = calc_finger_pose_in_gripper_frame(left_slider)
    X_GR = calc_finger_pose_in_gripper_frame(right_slider)

    M_LGo_G = calc_finger_spatial_inertia_in_gripper_frame(M_LLo_L, X_GL)
    M_RGo_G = calc_finger_spatial_inertia_in_gripper_frame(M_RRo_R, X_GR)

    M_CGo_G = M_GGo_G
    M_CGo_G += M_LGo_G
    M_CGo_G += M_RGo_G

    return M_CGo_G