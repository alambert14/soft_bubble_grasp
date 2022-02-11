import numpy as np
import os
import meshcat

import open3d as o3d
from manipulation.meshcat_utils import draw_open3d_point_cloud, draw_points
from manipulation.open3d_utils import create_open3d_point_cloud
from manipulation.utils import AddPackagePaths

import pydrake.common
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConnectMeshcatVisualizer,
    Diagram, DiagramBuilder,
    Parser, LoadModelDirectives, ProcessModelDirectives,
    RigidTransform, RotationMatrix,
)


def add_package_paths_local(parser: Parser):
    models_dir = os.path.join(os.curdir, 'models')
    iiwa_controller_models_dir = os.path.join(
        os.path.dirname(os.curdir), 'models')

    parser.package_map().Add(
        "drake_manipulation_models",
        os.path.join(pydrake.common.GetDrakePath(),
                     "manipulation/models"))
    parser.package_map().Add("local", models_dir)
    parser.package_map().Add('iiwa_controller',
                             iiwa_controller_models_dir)
    parser.package_map().PopulateFromFolder(models_dir)


class GraspSampler:
    def __init__(self, diagram: Diagram):
        self.rng = np.random.default_rng(seed=12345)
        self.base_diagram = diagram

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=1e-3
        )
        parser = Parser(self.plant)
        AddPackagePaths(parser)
        add_package_paths_local(parser)
        ProcessModelDirectives(
            LoadModelDirectives(
                os.path.join(os.curdir,
                             'models/camera_bins_gripper.yml')),
            self.plant, parser)
        self.plant.Finalize()

        self.viz = ConnectMeshcatVisualizer(builder,
                                            self.scene_graph,
                                            zmq_url='tcp://127.0.0.1:6000')
        self.viz.load()
        self.diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.Publish(context)
        # hide the gripper
        self.viz.vis["planning/plant/gripper"].set_property('visible', False)

    def generate_grasp_candidate_antipodal(self, plant_context,
                                           pcl, scene_graph_context):
        """
        Picks a random point in the cloud and aligns the bubble finger with the normal
        of that pixel. Tries a couple different alignments of the bubble gripper in
        order to account for the compliance.
        """
        tries = 100
        normal = None
        for i in range(tries):
            index = self.rng.integers(0, len(pcl.points) - 1)  # pick random point
            p_P = np.asarray(pcl.points[index])
            n_P = np.asarray(pcl.normals[index])
            n_P_norm = np.linalg.norm(n_P)

            # If the normal is close to 1, we've found a good point
            if np.isclose(n_P_norm, 1.0, atol=1e-2):
                normal = n_P / n_P_norm
                break

        if normal is None:
            raise RuntimeError(f'Cannot find point with good normal in {tries} tries.')

        # Insert meshcat visualization if needed

        gripper_y = normal
        # check if the normal is pointed down (positive x)
        normal_tol = 0.1
        if np.abs(np.dot(np.array([1, 0, 0]), gripper_y)) < normal_tol:
            return None  # reject

        gripper_x = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), gripper_y) * gripper_y
        gripper_z = np.cross(gripper_x, gripper_y)
        R_WG = RotationMatrix(np.vstack((gripper_x, gripper_y, gripper_z)).T)
        # How to generate this from the sdf file
        # Also will vary in x if we try to adjust the distance along the gripper
        p_GP = np.array([0.1, 0.05, 0])  # rough guess

        # Try different x positions
        min_dx = 0
        max_dx = 1.2
        step = 0.2
        body = self.plant.GetBodyByName('body')
        costs = []
        X_Gs = []
        for dx in range(int((max_dx - min_dx) / step)):
            # Next another for loop for rotation adjustments as necessary
            adjust = min_dx + dx * step  # make sure this is right
            p_GP2 = p_GP + np.array([adjust, 0., 0.])

            X_G = RigidTransform(R_WG, p_GP2)
            # Imagine the gripper is placed there
            self.plant.SetFreeBodyPose(plant_context, body, X_G)
            cost = self.grasp_candidate_cost(plant_context, pcl, scene_graph_context)

            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if not costs:
            return np.inf, None

        best_cost_index = np.asarray(costs).argsort()[0]
        best_X_G = X_Gs[best_cost_index]

        return best_cost_index, best_X_G

    def score_grasp_candidate(self, plant_context, pcl,
                              scene_graph_context):
        gripper_body = self.plant.GetBodyByName('bubble')  # why is this "body" in the example?
        X_G = self.plant.GetFreeBodyPose(plant_context, gripper_body)

        # Transform point cloud to gripper frame
        p_GC = X_G.inverse().multiply(np.asarray(pcl.points).T)

        query_object = self.scene_graph.get_query_output_port().Eval(scene_graph_context)
        # Check collisions between the robot(?) and the sink (or just the gripper?)
        if query_object.HasCollisions():
            # Infinite cost
            return np.inf

        for pt in pcl.points:
            pass

    def process_point_cloud(self, env_diagram, env_context, cameras, bin_name):
        env_plant = env_diagram.GetSubsystemByName("plant")
        env_plant_context = env_plant.GetMyContextFromRoot(env_context)

        # Compute crop box.
        bin_instance = env_plant.GetModelInstanceByName(bin_name)
        bin_body = env_plant.GetBodyByName("bin_base", bin_instance)
        X_B = env_plant.EvalBodyPoseInWorld(env_plant_context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply(
            [-.22 + 0.025 + margin, -.29 + 0.025 + margin, 0.015 + margin])
        b = X_B.multiply([.22 - 0.1 - margin, .29 - 0.025 - margin, 2.0])
        crop_min = np.minimum(a, b)
        crop_max = np.maximum(a, b)

        # Evaluate the camera output ports to get the images.
        merged_pcd = o3d.geometry.PointCloud()
        for c in cameras:
            point_cloud = env_diagram.GetOutputPort(f"{c}_point_cloud").Eval(env_context)
            pcd = create_open3d_point_cloud(point_cloud)

            # Crop to region of interest.
            pcd = pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,
                                                    max_bound=crop_max))
            if pcd.is_empty():
                continue

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))

            camera = env_plant.GetModelInstanceByName(c)
            body = env_plant.GetBodyByName("base", camera)
            X_C = env_plant.EvalBodyPoseInWorld(env_plant_context, body)
            pcd.orient_normals_towards_camera_location(X_C.translation())

            # Merge point clouds.
            merged_pcd += pcd

        # Voxelize down-sample.  (Note that the normals still look reasonable)
        return merged_pcd.voxel_down_sample(voxel_size=0.005)

    def grasp_candidate_cost(self, plant_context, pcl,
                             scene_graph_context):
        """
        Score based on amount of contact, non-collisions, and angle of the
        gripper normal
        """
        body = self.plant.GetBodyByName('body')
        X_G = self.plant.GetFreeBodyPose(plant_context, body)

        query_object = self.scene_graph.get_query_output_port().Eval(scene_graph_context)
        if query_object.HasCollisions():
            return np.inf  # infinite cost for impossible grasp

        # Transform point cloud to gripper frame
        p_GC = X_G.inverse().multiply(np.asarray(pcl.points).T)

        # Crop to inside the gripper
        crop_min = [-.05, 0.1, -0.00625]
        crop_max = [.05, 0.1125, 0.00625]
        indices = np.all((crop_min[0] <= p_GC[0, :], p_GC[0, :] <= crop_max[0],
                          crop_min[1] <= p_GC[1, :], p_GC[1, :] <= crop_max[1],
                          crop_min[2] <= p_GC[2, :], p_GC[2, :] <= crop_max[2]),
                         axis=0)

        # Check gripper/pointcloud collisions
        for pt in pcl.points:
            signed_distances = query_object.ComputerSignedDistanceToPoint(pt,
                                                                          0.0)
            if signed_distances:
                return np.inf

        # Score based on contact amount and also not too horizontal
        n_GC = X_G.rotation().multiply(np.asarray(pcl.normals)[indices, :].T)
        # Penalize deviation of the gripper from vertical.
        # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
        cost = 20.0 * X_G.rotation().matrix()[2, 1]

        # Reward sum |dot product of normals with gripper x|^2
        cost -= np.sum(n_GC[0, :] ** 2)

        return cost

    def find_best_grasps(self, env_context):

        pcl = self.process_point_cloud(self.base_diagram, env_context,
                                       ['camera0', 'camera1', 'camera2'], 'bin0')

        if pcl.is_empty():
            return []

        # create new context for ourdiagram
        context = self.diagram.CreateDefaultContext
        plant_context = self.plant.GetMyContextFromRoot(context)
        scene_graph_context = self.scene_graph.GetMyContextFromRoot(context)

        all_costs = []
        X_Gs = []

        for i in range(100):
            cost, X_G = self.generate_grasp_candidate_antipodal(
                plant_context, pcl,
                scene_graph_context
            )
            if np.isfinite(cost):
                all_costs.append(cost),
                X_Gs.append(X_G)

        # Take the indices of the 5 lowest costs
        best_cost_indices = np.asarray(all_costs).argsort()[:5]

        best_X_Gs = []
        for idx in best_cost_indices:
            best_X_Gs.append(X_Gs[idx])

        return best_X_Gs
