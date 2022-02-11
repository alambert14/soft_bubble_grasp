import os
import numpy as np

import graphviz
import pydrake.common
from pydrake.all import (
    AddMultibodyPlantSceneGraph, MeshcatVisualizerCpp, DiagramBuilder, Integrator,
    LeafSystem, MultibodyPlant, PiecewisePose, Quaternion, RigidTransform,
    RotationMatrix, SceneGraph, Simulator, TrajectorySource, MeshcatVisualizerParams, ConstantVectorSource,
    ConnectMeshcatVisualizer, InverseDynamicsController, Parser, ProcessModelDirectives, LoadModelDirectives,
    ZeroOrderHold, PidController, MeshcatContactVisualizer, ConnectContactResultsToDrakeVisualizer,
    Joint,
)
from pydrake.math import RollPitchYaw
from pydrake.examples.manipulation_station import ManipulationStation
# from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad
import meshcat

from create_plant import create_iiwa_soft_bubble_plant
from sample_grasps import GraspSampler
from manipulation.utils import AddPackagePaths
from manipulation.scenarios import AddRgbdSensors

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


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


class BubbleGripperSystem:

    def __init__(self):
        builder = DiagramBuilder()

        # self.station = ManipulationStation()
        # self.station.SetupClutterClearingStation()
        # self.station.Finalize()

        self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-4)
        parser = Parser(self.plant)
        AddPackagePaths(parser)  # Russ's manipulation repo.
        add_package_paths_local(parser)  # local.
        directive_file = os.path.join(
            os.getcwd(), 'models', 'iiwa_bubble_and_two_bins.yml')

        ProcessModelDirectives(LoadModelDirectives(directive_file), self.plant, parser)

        # Add cucumber
        cucumber_sdf = os.path.join(os.curdir, 'models', 'cucumber_simplified.sdf')
        cucumber_model = parser.AddModelFromFile(cucumber_sdf, 'cucumber')
        self.plant.Finalize()

        # Add cameras 0, 1, 2
        AddRgbdSensors(builder, self.plant, scene_graph)

        plant_iiwa_controller, _ = create_iiwa_soft_bubble_plant(
            self.plant.gravity_field().gravity_vector()
        )
        Kp_iiwa = np.array([800, 600, 600, 400, 400, 400, 200])
        Kd_iiwa = 2 * np.sqrt(Kp_iiwa)
        Ki_iiwa = np.zeros(7)
        idc = InverseDynamicsController(plant_iiwa_controller, Kp_iiwa,
                                        Ki_iiwa, Kd_iiwa, False)
        builder.AddSystem(idc)
        model_iiwa = self.plant.GetModelInstanceByName('iiwa')
        # Connect the inputs and outputs of the controller after adding the controller
        # to the system
        builder.Connect(self.plant.get_state_output_port(model_iiwa),
                        idc.get_input_port_estimated_state())
        builder.Connect(idc.get_output_port_control(),
                        self.plant.get_actuation_input_port(model_iiwa))

        q0 = ConstantVectorSource([0., 0., 0., -1.57, 0., 1.57, 0., 0, 0, 0, 0, 0, 0, 0])
        builder.AddSystem(q0)
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url='tcp://127.0.0.1:6000', prefix="environment")

        # Contact visualization?
        # contact_viz = MeshcatContactVisualizer(
        #     meshcat_viz=viz,
        #     force_threshold=0,
        #     contact_force_scale=1,
        #     plant=self.plant,
        #     contact_force_radius=0.005,
        # )
        # contact_input_port = contact_viz.GetInputPort('contact_results')
        # ConnectContactResultsToDrakeVisualizer(
        #     builder,
        #     self.plant,
        #     scene_graph,
        # )
        # builder.Connect(
        #     self.plant.GetOutputPort('contact_results'),
        #     contact_input_port,
        # )

        builder.Connect(q0.GetOutputPort('y0'), idc.get_input_port_desired_state())

        # Add the bubble gripper
        model_bubble = self.plant.GetModelInstanceByName('bubble')
        Kp_bubble = np.array([5, 5])
        Ki_bubble = np.zeros(2)
        Kd_bubble = np.array([0.5, 0.5])
        pid = PidController(Kp_bubble, Ki_bubble, Kd_bubble)
        pid.set_name('gripper_finger_controller')
        builder.AddSystem(pid)
        builder.Connect(
            pid.get_output_port_control(),
            self.plant.get_actuation_input_port(model_bubble))
        builder.Connect(
            self.plant.get_state_output_port(model_bubble),
            pid.get_input_port_estimated_state())

        bubble_desired_state = ConstantVectorSource([-0.05, 0.05, 0, 0])
        builder.AddSystem(bubble_desired_state)
        builder.Connect(
            bubble_desired_state.GetOutputPort('y0'),
            pid.get_input_port_desired_state())



        self.diagram = builder.Build()
        render_system_with_graphviz(self.diagram)

        # builder.AddSystem(self.station)
        # self.plant = self.station.get_mutable_multibody_plant()
        # print(self.station.GetGraphvizString())

        # gripper_position = builder.AddSystem(ConstantVectorSource([0.1]))

        # builder.Connect(gripper_position.get_output_port(), self.station.GetInputPort('wsg_position'))


        # self.gripper_frame = self.plant.GetFrameByName('body')  # base frame of iiwa?
        # self.world_frame = self.plant.world_frame()

        self.context = self.diagram.CreateDefaultContext()
        # passing in the object here?
        self.plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.context)

        # Move the cucumber
        bin_inst = self.plant.GetModelInstanceByName('bin0')
        bin_body = self.plant.GetBodyByName('bin_base', bin_inst)
        X_B = self.plant.EvalBodyPoseInWorld(self.plant_context, bin_body)
        X_BDrop = RigidTransform(np.array([0., 0., 0.4]))
        X_Drop = X_B.multiply(X_BDrop)
        cucumber_obj = self.plant.GetBodyByName('base_link', cucumber_model)

        self.plant.SetFreeBodyPose(self.plant_context, cucumber_obj, X_Drop)



        # station_context = self.diagram.GetMutableSubsystemContext(self.station, context)

        # q_0 = np.array([0., 0., 0, -1.57, 0, 1.57, 0])

        # self.station.SetIiwaPosition(station_context, q_0)
        # self.station.SetIiwaVelocity(station_context, np.zeros(7))
        # self.station.SetWsgPosition(station_context, 0.1)
        # self.station.SetWsgVelocity(station_context, 0)

        self.diagram.Publish(self.context)
        self.viz = self.diagram.GetSubsystemByName('meshcat_visualizer')

    def start(self):
        self.viz.reset_recording()
        self.viz.start_recording()

        # station_context = self.diagram.GetMutableSubsystemContext(self.station, context)
        # q_0 = np.array([0., 0., 0, -1.57, 0, 1.57, 0])

        # self.station.SetIiwaPosition(station_context, q_0)
        # self.station.SetIiwaVelocity(station_context, np.zeros(7))
        # self.station.SetWsgPosition(station_context, 0.1)
        # self.station.SetWsgVelocity(station_context, 0)

        simulator = Simulator(self.diagram, self.context)
        simulator.set_target_realtime_rate(1.0)

        duration = 1
        simulator.AdvanceTo(duration)
        self.viz.stop_recording()
        self.viz.publish_recording()
        print('creating grasp sampler')
        grasp_sampler = GraspSampler(self.diagram)
        print(grasp_sampler.find_best_grasps(self.context))

if __name__ == '__main__':
    # meshcat_ = StartMeshcat()
    v = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    v.delete()

    robot = BubbleGripperSystem()

    robot.start()
