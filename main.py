import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph, MeshcatVisualizerCpp, DiagramBuilder, Integrator,
    LeafSystem, MultibodyPlant, PiecewisePose, Quaternion, RigidTransform, RollPitchYaw,
    RotationMatrix, SceneGraph, Simulator, TrajectorySource, MeshcatVisualizerParams, ConstantVectorSource,
    ConnectMeshcatVisualizer,
)
from pydrake.examples.manipulation_station import ManipulationStation
# from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad

import meshcat

class BubbleGripperManipulator():

    def __init__(self):
        builder = DiagramBuilder()

        self.station = ManipulationStation()
        self.station.SetupClutterClearingStation()
        self.station.Finalize()

        builder.AddSystem(self.station)
        self.plant = self.station.get_mutable_multibody_plant()
        # builder.AddSystem(self.station.get_scene_graph())
        # not sure how to get scene graph
        viz = ConnectMeshcatVisualizer(
            builder, self.station.get_scene_graph(), zmq_url='tcp://127.0.0.1:6000', prefix="environment")

        # params = MeshcatVisualizerParams()
        # params.delete_on_initialization_event = False
        # self.visualizer = MeshcatVisualizerCpp.AddToBuilder(
        #     builder, self.station.GetOutputPort('query_object'), self.viz, params)


        gripper_position = builder.AddSystem(ConstantVectorSource([0.1]))

        builder.Connect(gripper_position.get_output_port(), self.station.GetInputPort('wsg_position'))


        self.diagram = builder.Build()
        self.gripper_frame = self.plant.GetFrameByName('body')  # base frame of iiwa?
        self.world_frame = self.plant.world_frame()

        context = self.diagram.CreateDefaultContext()
        # passing in the object here?
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)
        station_context = self.diagram.GetMutableSubsystemContext(self.station, context)

        q_0 = np.array([0., 0., 0, -1.57, 0, 1.57, 0])

        self.station.SetIiwaPosition(station_context, q_0)
        self.station.SetIiwaVelocity(station_context, np.zeros(7))
        self.station.SetWsgPosition(station_context, 0.1)
        self.station.SetWsgVelocity(station_context, 0)

        self.diagram.Publish(context)


        self.viz = self.diagram.GetSubsystemByName('meshcat_visualizer')

    def start(self):
        self.viz.reset_recording()
        self.viz.start_recording()
        context = self.diagram.CreateDefaultContext()
        station_context = self.diagram.GetMutableSubsystemContext(self.station, context)
        q_0 = np.array([0., 0., 0, -1.57, 0, 1.57, 0])

        self.station.SetIiwaPosition(station_context, q_0)
        self.station.SetIiwaVelocity(station_context, np.zeros(7))
        self.station.SetWsgPosition(station_context, 0.1)
        self.station.SetWsgVelocity(station_context, 0)

        simulator = Simulator(self.diagram, context)
        simulator.set_target_realtime_rate(1.0)

        duration = 10
        simulator.AdvanceTo(duration)
        self.viz.stop_recording()
        self.viz.publish_recording()


if __name__ == '__main__':
    # meshcat_ = StartMeshcat()
    v = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    v.delete()

    robot = BubbleGripperManipulator()

    robot.start()

