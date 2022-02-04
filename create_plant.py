import os
import numpy as np

import pydrake.common
from pydrake.all import (
    MultibodyPlant, Parser, ProcessModelDirectives, LoadModelDirectives,
    SpatialInertia, FindResourceOrThrow, RigidTransform,
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

    gripper_plant = MultibodyPlant(1e-3)
    gripper_parser = Parser(gripper_plant)
    gripper_parser.AddModelFromFile(bubble_sdf_file)
    gripper_plant.Finalize()

    iiwa_model = plant.GetModelInstanceByName('iiwa')
    bubble_gripper = plant.AddRigidBody(
        'bubble', iiwa_model, SpatialInertia())
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