
# Quantum Internet Project, written using netsquid
# See netsquid documentation for details:
#   https://docs.netsquid.org/latest-release/
#
#  Repeater chain tutorial show an alternative solution:
#   https://docs.netsquid.org/latest-release/learn_examples/learn.examples.repeater_chain.html
import netsquid as ns
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node, Connection, Network, DirectConnection
from netsquid.protocols.protocol import Signals
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qchannel import QuantumChannel
from netsquid.qubits import qubitapi
from netsquid.qubits import ketstates
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.components.models import FibreLossModel
from netsquid.util.datacollector import DataCollector
from netsquid.components import Message
import netsquid as ns
import pydynaa
from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi
from netsquid.components import instructions as instr

from netsquid.components import INSTR_INIT, INSTR_H, INSTR_X, INSTR_Z, INSTR_CNOT, INSTR_MEASURE, INSTR_MEASURE_BELL

import math

from pydynaa import EventType


"""
Nework

    +---------+                                           +------+                                           +-------+
    |  Start  |----- Quantum Channel generating Qbits ----| Swap |----- Quantum Channel generating Qbits ----| End   |
    |  node   |                                           | Node |                                           | node  |
    |         |----- Classical Channel: control data  ----|      |----- Classical Channel: control data  ----|       |
    |  Alice  |                                           |      |                                           |  Bob  |
    +---------+                                           +------+                                           +-------+

Core task:
    create an "entangled channel" connecting Alice with Bob
        (by performing entanglement swapping in the central node)
    entanglement fidelity is printed for every entanglement pair
"""

class EntanglingConnection(Connection):
    """
    Connection ussed to generate a pair of entangled Qubits
    Inspired by examples shown during seminars
    """
    def __init__(self, p_m, p_lr, t_clock, length, name="EntanglingConnection"):
        self.properties['length'] = length

        super().__init__(name=name)
        # PHOTON SOURCE - initially disabled, starts when remotely controlled
        qsource = QSource(f"EPS", StateSampler([ks.b00, None], [p_m, 1-p_m]),
                          num_ports=2,
                          timing_model=FixedDelayModel(delay=t_clock),
                          status=SourceStatus.OFF)

        self.add_subcomponent(qsource, name=qsource.name)

        # CARACTERIZE NOISE forms
        models = {
            "delay_model":
                FibreDelayModel(),
            "quantum_loss_model":
                FibreLossModel(p_loss_init=1 - p_lr, p_loss_length=0.),
            "quantum_noise_model":
                DepolarNoiseModel(depolar_rate=0.1, time_independent=True)
        }

        # DEFINE QUANTUM CHANNELS
        # connect to the node at the left (Alice or middleware)
        qchannel_left = QuantumChannel("qchannel_left", length=length / 2,
                                       models=models)
        # connect to the node at the tight (middleware or Bob)
        qchannel_right = QuantumChannel("qchannel_right", length=length / 2,
                                        models=models)
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_left, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_right, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_left.ports["send"])
        qsource.ports["qout1"].connect(qchannel_right.ports["send"])


# protocol handling the source of entangled photons
# Protocol mechanism:
# - K photons are contecutively generated
# - only the first reeceived of them is effectively stored
# - we hope captured photons are from the same pair
# - otherwise repeat 
#
# Protocol execute on network devices
class MidpointSourceProtocol(NodeProtocol):
    """
    Midpoint Source Protocol
    two actor: left and right actor (in the drawning)
    left actor (active):
        start the quantum channel, "active role".
    right actor:
        "passive".
    See seminars.
    """
    ENTANGLED_SIGNAL = "new_entangled_pair"
    ENTANGLED_EVT_TYPE = EventType(ENTANGLED_SIGNAL, "New entangled pair generated")

    # Procol parameters:
    #   node:         node to be controlled
    #   K_attempts:  number of consequently generated
    #   t_clock:      perriodbeetwen two consecutive generated photons
    #   link_length:  length of the connection, necessary for temporization
    #                   problems
    #   connection:   the connection of the generating photon source
    #                   is a trick to control EPS inside it
    #   active:       should this node activate and deactivate the
    #                   photon source?
    #   mem_position: memory cell to operate on - used because multiple cell
    #                   could be theoretically employed
    #   qport:        name of the quantum port to be used to receive
    #                   qubits
    #   cport:        name of the classical port to be used to handle
    #                   exchange control messages
    def __init__(self, node, K_attempts, t_clock, link_length, connection=None, active=False, mem_position=0, qport=None, cport=None):
        super().__init__(node, name="MSProtocol_{mem_position}")
        self.K_attempts = K_attempts
        self.t_clock = t_clock
        self.link_length = link_length
        self.connection = connection
        self.active = active
        self.mem_position = mem_position
        self.qport = qport
        self.cport = cport

        self.add_signal(MidpointSourceProtocol.ENTANGLED_SIGNAL,
                        MidpointSourceProtocol.ENTANGLED_EVT_TYPE)

    def run(self):
        # directly access classical and quantum port by name
        cport = self.node.ports[self.cport]
        qport = self.node.ports[self.qport]

        print(f"[{ns.sim_time()}] Node {self.node.name}: Starting MS Protocol instance")
        # compute the approximate time necessary to
        # have qubits traversing the connection
        # time is expressed is nanoseconds
        t_link = 1e9 * self.link_length / 200000 # 200Mm/s

        # start the EPS
        if self.active:
            # start and wait for the first qubit to synchronize
            # start EPS
            self.connection.subcomponents["EPS"].status = SourceStatus.INTERNAL
            print(f"[{ns.sim_time()}] Node {self.node.ID}: Starting EPS")
            # wait first qubit
            yield self.await_port_input(qport)
        
        # send message to start the twin instance
        if self.active:
            # current time is "aligned" with EPS gen rate
            start_time = math.ceil(ns.sim_time() + t_link)
            # round up to a few nanoseconds before next clock cycle
            start_time = (start_time - start_time % self.t_clock) + 2*self.t_clock - 1
            print(f"[{ns.sim_time()}] Node {self.node.ID}: Sending START message with value {start_time}")
            cport.tx_output(Message(items=["START", start_time]))
        # or wait for the sent message
        else:
            # wait for the other node to tell the starting time
            print(f"[{ns.sim_time()}] Node {self.node.ID}: Waiting for START message")
            yield self.await_port_input(cport)
            msg = cport.rx_input().items
            # ensure it is a START message
            assert msg[0] == "START"
            print(f"[{ns.sim_time()}] Node {self.node.ID}: Received START message")
            # then extract time to wait
            start_time = msg[1]

        # anyway, wait until the chosen (or received) time
        # rely on "global time"
        yield self.await_timer(end_time=start_time)

        # now both devices are ready to start!
        # maximum time to wait for a qubit - take a 1us for safety
        t_round = self.K_attempts*self.t_clock + 1000

        success_index = None

        print(f"[{ns.sim_time()}] Node {self.node.ID}: Starting entanglement generation")

        while True:
            # wait for the first photon to arrive - or timeout
            ev_expr = yield self.await_port_input(qport) | self.await_timer(end_time=start_time+t_round)

            # check first photon arrival
            if ev_expr.first_term.value and success_index is None:
                # receive the photon
                qubit = qport.rx_input().items[0]
                # store the qbit in memory
                self.node.qmemory.put(qubit, positions=[self.mem_position])
                
                # compute current index ('round' might be better)
                success_index = math.floor((ns.sim_time() - start_time) / self.t_clock)
                print(f"[{ns.sim_time()}] Node {self.node.ID}: Latched photon at attempt {success_index}")

            # timeout?
            if ev_expr.second_term.value:
                # if no qubit was captured, set -1 as fail value
                if success_index is None:
                    success_index = -1
                # transmit captured qbits index 
                final_msg = Message(items=["END", success_index])
                cport.tx_output(final_msg)
                # wait for the qubits sent by the other device
                yield self.await_port_input(cport)
                recv_end_msg = cport.rx_input().items
                # assert protocol synchronization
                assert recv_end_msg[0] == "END"

                # check convergence
                if recv_end_msg[1] != -1 and recv_end_msg[1] == success_index:
                    print(f"[{ns.sim_time()}] Node {self.node.ID}: Entanglement generation successful"
                          f" at attempt {success_index}")
                    # signal the success - used to communicate with caller protocol
                    self.send_signal(self.ENTANGLED_SIGNAL, result=None)

                    # the EPS can be stopped
                    if self.active:
                        self.connection.subcomponents["EPS"].status = SourceStatus.OFF
                    # end of the procol - return to the caller
                    return

                # otherwise try again
                else:
                    # unlucky, try again
                    print(f"[{ns.sim_time()}] Node {self.node.ID}: Entanglement generation failed."
                          f" Starting new round")
                    # simulation does not suffer throttling - we can use exact time
                    start_time = ns.sim_time()
                    success_index = None
                    # eventually remove qbit from memory
                    if self.mem_position in self.node.qmemory.used_positions:
                        self.node.qmemory.pop(positions=self.mem_position)

# entanglement swapping protocol - local instance
class EntanglementSwappingProtocol(NodeProtocol):
    """
    Perform entannglement swapping:
    see seminars
    """

    SWAPPING_SIGNAL = "entanglement_swapped"
    SWAPPING_EVT_TYPE = EventType(SWAPPING_SIGNAL, "Entanglement swapped!")

    # Protocol instance parameters:
    # - node:           node executing protocol
    # - mem_positions:  2 option
    #                       - int|Tuple[int] :-> wait for data and perform
    #                                            corrections
    #                       - Tuple[int,int] :-> perform measure and
    #                                            correction data, wait for
    #                                            acknowledgement
    # - cport:          classical port to wait input for
    # - name:           name of the protocol instance
    #
    # To be set:
    #   active: True if the node performs the measures
    #           False if the node performs the corrections
    def __init__(self, node, mem_positions, cport=None, name=None):
        super().__init__(node, name)
        self.cport = cport

        if isinstance(mem_positions, int):
            self.mem_positions = [mem_positions]
            self.active = False
        elif isinstance(mem_positions, tuple) and len(mem_positions) == 1 and \
            isinstance(mem_positions[0], int):
            self.mem_positions = list(mem_positions)
            self.active = False
        elif isinstance(mem_positions, tuple) and len(mem_positions) == 2 and \
            isinstance(mem_positions[0], int) and isinstance(mem_positions[1], int):
            self.mem_positions = list(mem_positions)
            self.active = True
        else:
            raise TypeError(f"Bad input: 'mem_positions' ({mem_positions})")

        self.add_signal(EntanglementSwappingProtocol.SWAPPING_SIGNAL,
                        EntanglementSwappingProtocol.SWAPPING_EVT_TYPE)

        if self.active:
            # teleport and Bell Measurement
            class TeleportationProgram(QuantumProgram):
                default_num_qubits = 2
                def program(self):
                    q0,q1 = self.get_qubit_indices(2)
                    self.apply(instruction=instr.INSTR_CNOT, qubit_indices=[q0, q1])
                    self.apply(instruction=instr.INSTR_H, qubit_indices=[q0])
                    self.apply(instruction=instr.INSTR_MEASURE, qubit_indices=[q0], output_key="m1")
                    self.apply(instruction=instr.INSTR_MEASURE, qubit_indices=[q1], output_key="m2")
                    yield self.run()

            self.measure_prog = TeleportationProgram()
        else:
            class CorrectionProgram(QuantumProgram):
                default_num_qubits = 1
                def __init__(self):
                    super().__init__(num_qubits=None, parallel=True, qubit_mapping=None)
                def set_corrections(self, m1, m2):
                    self.m1 = m1
                    self.m2 = m2
                def program(self):
                    q0 = self.get_qubit_indices(1)
                    # Do corrections:
                    if self.m2:
                        self.apply(instr.INSTR_X, q0)
                    if self.m1:
                        self.apply(instr.INSTR_Z, q0)
                    yield self.run()

            self.correction_prog = CorrectionProgram()

    def run(self):
        cport = self.node.ports[self.cport]
        qproc = self.node.qmemory

        print(f"[{ns.sim_time()}] Node {self.node.ID}: Starting entanglement swapping protocol")

        if self.active:
            # active role, perform measures and send data
            # operate with CNOT
            # operate with H first qbit
            # perform measures
            qproc.execute_program(self.measure_prog, qubit_mapping=[0, 1], error_on_fail=True)
            yield self.await_program(qproc)
            # get results
            m1 = self.measure_prog.output["m1"][0]
            m2 = self.measure_prog.output["m2"][0]

            print(f"[{ns.sim_time()}] Node {self.node.ID}: performed measures, m1 = {m1}, m2 = {m2}")

            # send data via classical channel
            cport.tx_output(Message(items=["SWAP_CORRECTIONS", [m1, m2]]))
            # wait for ack
            yield self.await_port_input(cport)
            recv_end_msg = cport.rx_input().items
            assert recv_end_msg[0] == "SWAP_ACK"
            print(f"[{ns.sim_time()}] Node {self.node.ID}: SWAP_ACK received")
        else:
            # wait for correction data on classical channel
            yield self.await_port_input(cport)
            recv_end_msg = cport.rx_input().items
            assert recv_end_msg[0] == "SWAP_CORRECTIONS"

            print(f"[{ns.sim_time()}] Node {self.node.ID}: SWAP_CORRECTIONS received")
            m1, m2 = recv_end_msg[1]
            # perform correction
            self.correction_prog.set_corrections(m1, m2)
            qproc.execute_program(self.correction_prog, qubit_mapping=[0], error_on_fail=True)
            yield self.await_program(qproc)

            print(f'[{ns.sim_time()}] Node {self.node.ID}: performed correction = {"X" if m2 else ""}{"Z" if m1 else ""}{"None" if m1+m2==0 else ""}')

            # send ack
            final_msg = Message(items=["SWAP_ACK"])
            cport.tx_output(final_msg)
            print(f"[{ns.sim_time()}] Node {self.node.ID}: SWAP_ACK sent")


        # protocol as terminated, send termination signal
        self.send_signal(self.SWAPPING_SIGNAL, result=None)


class ClassicalConnection(DirectConnection):
    """A connection that transmits classical messages in two ways"""

    def __init__(self, link_length, name):
        channel_L_to_R = ClassicalChannel("channel_L_to_R", length=link_length,
                                           models={"delay_model": FibreDelayModel()})
        channel_R_to_L = ClassicalChannel("channel_R_to_L", length=link_length,
                                           models={"delay_model": FibreDelayModel()})
        super().__init__(name=name,
                         channel_AtoB=channel_L_to_R,
                         channel_BtoA=channel_R_to_L)



class Alice(Node):
    """First node in the line"""
    def __init__(self, ID, name):
        super().__init__(name=name, ID=ID, port_names=[
            # right ports are connected to the Swapper
            "RIGHT_q0", "RIGHT_c0"
        ])

        physical_instructions = [
            PhysicalInstruction(INSTR_INIT, duration=1., parallel=True),
            PhysicalInstruction(INSTR_H, duration=1., parallel=True),
            PhysicalInstruction(INSTR_CNOT, duration=1., parallel=True),
            PhysicalInstruction(INSTR_MEASURE, duration=1., parallel=True)
        ]
        # two bits qprocessor - contain qbit to be teleported and entangled one
        qproc = QuantumProcessor("qproc",
                                 num_positions=1,
                                 phys_instructions=physical_instructions)
        self.qmemory = qproc

class Bob(Node):
    """Last node in the line"""
    def __init__(self, ID, name):
        super().__init__(name=name, ID=ID, port_names=[
            # left ports are connected to the Swapper
            "LEFT_q0", "LEFT_c0"
        ])

        physical_instructions = [
            PhysicalInstruction(INSTR_X, duration=1., parallel=True),
            PhysicalInstruction(INSTR_Z, duration=1., parallel=True),
            PhysicalInstruction(INSTR_MEASURE, duration=1., parallel=True)
        ]
        # single bit qprocessor - contain only final qubit
        qproc = ns.components.QuantumProcessor("qproc", num_positions=1, phys_instructions=physical_instructions)
        self.qmemory = qproc

class Swapper(Node):
    """This class implements a quantum repeater node"""
    def __init__(self, ID, name):
        super().__init__(name=name, ID=ID, port_names=[
            # left ports are connected to Alice
            "LEFT_q0", "LEFT_c0"
            # right ports are connected to Bob
            "RIGHT_q0", "RIGHT_c0"
        ])
        # self.qmemory = ns.components.QuantumMemory("qmemory", num_positions=2)

        physical_instructions = [
            PhysicalInstruction(INSTR_H, duration=1., parallel=True),
            PhysicalInstruction(INSTR_CNOT, duration=1., parallel=True),
            PhysicalInstruction(INSTR_MEASURE, duration=1., parallel=True)
        ]
        qproc = ns.components.QuantumProcessor("qproc", num_positions=2, phys_instructions=physical_instructions)

        self.qmemory = qproc

class AliceProtocol(NodeProtocol):
    """Controls Alice"""
    def __init__(self, node, K_attempts, link_length, t_clock, connection):
        super().__init__(node, name="AliceProtocol")
        self.add_subprotocol(MidpointSourceProtocol(
            node=node, K_attempts=K_attempts, t_clock=t_clock, link_length=link_length,
            connection=connection, active=True, mem_position=0,
            qport="RIGHT_q0", cport="RIGHT_c0"), name="MSP_0")

    # Fidelity is measured for entangled Bell states
    def _get_fidelity(self, position):
        qubits = self.node.qmemory.peek(positions=[position])[0].qstate.qubits
        fidelity = qubitapi.fidelity(qubits, ketstates.b00, squared=True)
        return fidelity

    def run(self):
        # the node is Alice - first of the chain
        alice = self.node
        # only one qbit to entangle
        MSP = self.subprotocols["MSP_0"]
        # classical port to wait for classical message
        cport = alice.ports["RIGHT_c0"]

        # entangle qubits by using MSP protocol
        MSP.start()
        # wait for its termination
        yield self.await_signal(sender=MSP, signal_label=MidpointSourceProtocol.ENTANGLED_SIGNAL)

        # get the fidelity of the qubits entangled with the Swapper
        fidelity_0 = self._get_fidelity(0)
        print(f"[{ns.sim_time()}] Alice-Swapper {alice.ID}: Qubits are entangled with fidelity {fidelity_0}")

        # at this point wait for swapper telling the other channel is set
        yield self.await_port_input(cport)
        assert cport.rx_input().items[0] == "OTHER_SET"
        print(f"[{ns.sim_time()}] Alice {alice.ID}: Other qchannel is ready for swapping!")

        # now we can wait for entangle swapping
        yield self.await_port_input(cport)
        assert cport.rx_input().items[0] == "SWAPPED"
        print(f"[{ns.sim_time()}] Alice {alice.ID}: swapping performed!")

        # last fidelity check!
        fidelity_e2e = self._get_fidelity(0)
        print(f"[{ns.sim_time()}] Alice-Bob {alice.ID}: Qubits are entangled with fidelity {fidelity_e2e}")


class BobProtocol(NodeProtocol):
    """Controls sBob"""
    def __init__(self, node, K_attempts, link_length, t_clock, connection):
        super().__init__(node, name="BobProtocol")
        self.add_subprotocol(MidpointSourceProtocol(
            node=node, K_attempts=K_attempts, t_clock=t_clock, link_length=link_length,
            connection=connection, active=False, mem_position=0,
            qport="LEFT_q0", cport="LEFT_c0"), name="MSP_1")

        self.add_subprotocol(EntanglementSwappingProtocol(node=node,
            mem_positions=(0), cport="LEFT_c0"), name="EnSwapCorr")


    def run(self):
        # the node is bob - first of the chain
        bob = self.node
        # only one qbit to entangle
        MSP = self.subprotocols["MSP_1"]
        ESCorr = self.subprotocols["EnSwapCorr"]
        # classical port to wait for classical message
        cport = bob.ports["LEFT_c0"]

        # entangle qubits by using MSP protocol
        MSP.start()
        # wait for its termination
        yield self.await_signal(sender=MSP, signal_label=MidpointSourceProtocol.ENTANGLED_SIGNAL)

        # now wait for the other channel to be set
        # at this point wait for swapper telling the other channel is set
        yield self.await_port_input(cport)
        assert cport.rx_input().items[0] == "OTHER_SET"
        print(f"[{ns.sim_time()}] Bob {bob.ID}: Other qchannel is ready for swapping!")

        # then wait for entangle swapping and relative corrections
        ESCorr.start()
        yield self.await_signal(sender=ESCorr, signal_label=EntanglementSwappingProtocol.SWAPPING_SIGNAL)
        print(f"[{ns.sim_time()}] Swapper {bob.ID}: Terminated all protocols")


# swapper handle two different instances of the protocol with both end nodes
class SwapperProtocol(NodeProtocol):
    """Controls Swapper"""
    def __init__(self, node, K_attempts, link_length, t_clock, connection_left, connection_right):
        super().__init__(node, name="BobProtocol")

        # instance interacting with Alice - it is passive
        self.add_subprotocol(MidpointSourceProtocol(
            node=node, K_attempts=K_attempts, t_clock=t_clock, link_length=link_length,
            connection=connection_left, active=False, mem_position=0,
            qport="LEFT_q0", cport="LEFT_c0"), name="MSP_0")

        # instance interacting with Bob - it is "active" (i.e. controls the EPS)
        self.add_subprotocol(MidpointSourceProtocol(
            node=node, K_attempts=K_attempts, t_clock=t_clock, link_length=link_length,
            connection=connection_right, active=True, mem_position=1,
            qport="RIGHT_q0", cport="RIGHT_c0"), name="MSP_1")

        self.add_subprotocol(EntanglementSwappingProtocol(node=node,
            mem_positions=(0,1), cport="RIGHT_c0"), name="EnSwapMeas")

    # Fidelity is measured for entangled Bell states
    def _get_fidelity(self, position):
        qubits = self.node.qmemory.peek(positions=[position])[0].qstate.qubits
        fidelity = qubitapi.fidelity(qubits, ketstates.b00, squared=True)
        return fidelity

    def run(self):
        # the node is Swapper - first of the chain
        swapper = self.node
        # only one qbit to entangle
        MSP_0 = self.subprotocols["MSP_0"]
        MSP_1 = self.subprotocols["MSP_1"]
        ESMeas = self.subprotocols["EnSwapMeas"]
        # classical port to wait for classical message
        cport_left = swapper.ports["LEFT_c0"]
        cport_right = swapper.ports["RIGHT_c0"]

        # starts both protcol instances
        MSP_0.start()
        MSP_1.start()
        # wait for their termination
        yield self.await_signal(sender=MSP_0, signal_label=MidpointSourceProtocol.ENTANGLED_SIGNAL) \
            & self.await_signal(sender=MSP_1, signal_label=MidpointSourceProtocol.ENTANGLED_SIGNAL)
        print(f"[{ns.sim_time()}] Swapper {swapper.ID}: Both connections entangled")

        # send both ends "OTHER_SET"
        self.node.ports["LEFT_c0"].tx_output(Message(items=["OTHER_SET"]))
        self.node.ports["RIGHT_c0"].tx_output(Message(items=["OTHER_SET"]))
        print(f"[{ns.sim_time()}] Swapper {swapper.ID}: send 'OTHER_SET' to Alice and Bob")


        # both entangling subprotocols are terminated
        # shoud calculate fidelity only on qubit entangled with Bob
        fidelity_1 = self._get_fidelity(1)
        print(f"[{ns.sim_time()}] Swapper-Bob {swapper.ID}: Qubits are entangled with fidelity {fidelity_1}")

        # perform entanglement swapping
        ESMeas.start()
        yield self.await_signal(sender=ESMeas, signal_label=EntanglementSwappingProtocol.SWAPPING_SIGNAL)
        print(f"[{ns.sim_time()}] Swapper {swapper.ID}: Terminated all protocols")

        self.node.ports["LEFT_c0"].tx_output(Message(items=["SWAPPED"]))
        print(f"[{ns.sim_time()}] Swapper {swapper.ID}: Send 'SWAPPED' to Alice")


# build topology
# Parameters:
# - link_length: length of connections between nodes
# - p_lr:        probabi
# - p_m:         EPS parameter, probability photons are generated
# - t_clock:     EPS parameter, time between two photon generation attempt
def get_network(link_length, p_lr, p_m, t_clock):
    # TOPOLOGY
    network = Network("Project_network")

    # create all three nodes
    alice = Alice(name="Alice", ID=1)
    swapper = Swapper(name="Swapper", ID=2)
    bob = Bob(name="Bob", ID=3)

    # add to the network
    network.add_nodes([alice, swapper, bob])

    # connect them
    # Alice - Swapper
    q1 = EntanglingConnection(t_clock=t_clock, length=link_length,
                              p_m=p_m, p_lr=p_lr, name="QCon_1")
    c1 = ClassicalConnection(link_length, "CCon_1")

    # Quantum connection
    # [alice]RIGHT_q0}===q1==={LEFT_q0[swapper]
    network.add_connection(
        alice, swapper, connection=q1, label="quantum_1",
        port_name_node1="RIGHT_q0", port_name_node2="LEFT_q0")
    # qubits are not automatically transferd to memory but
    # this is done via protocols (i.e. software)

    # Classical connection
    # [alice]RIGHT_c0}===c1==={LEFT_c0[swapper]
    network.add_connection(
        alice, swapper, connection=c1, label="classical_1",
        port_name_node1="RIGHT_c0", port_name_node2="LEFT_c0")

    # Swapper - Bob
    q2 = EntanglingConnection(t_clock=t_clock, length=link_length,
                              p_m=p_m, p_lr=p_lr, name="QCon_2")
    c2 = ClassicalConnection(link_length, "CCon_2")

    # Quantum connection
    # [swapper]RIGHT_q0}===q2==={LEFT_q0[bob]
    network.add_connection(
        swapper, bob, connection=q2, label="quantum_2",
        port_name_node1="RIGHT_q0", port_name_node2="LEFT_q0")

    # Classical connection
    # [swapper]RIGHT_c0}===c2==={LEFT_c0[bob]
    network.add_connection(
        swapper, bob, connection=c2, label="classical_2",
        port_name_node1="RIGHT_c0", port_name_node2="LEFT_c0")


    # determine number of EPS attemps
    K_attempts = math.ceil(1/(p_m*p_lr))

    # DEFINE PROTOCOLS
    proto_alice = AliceProtocol(alice, K_attempts=K_attempts, link_length=link_length,
                            t_clock=t_clock, connection=q1)
    proto_swapper = SwapperProtocol(swapper, K_attempts=K_attempts, link_length=link_length,
                            t_clock=t_clock, connection_left=q1, connection_right=q2)
    proto_bob = BobProtocol(bob, K_attempts=K_attempts, link_length=link_length,
                            t_clock=t_clock, connection=q2)

    proto_alice.start()
    proto_swapper.start()
    proto_bob.start()

    return network, proto_alice, proto_swapper, proto_bob



if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.DM)
    print("START simulation")

    network, *_ = get_network(link_length=1e-3, p_lr=0.9, p_m=0.02, t_clock=10)

    stats = ns.sim_run(end_time=10000000)

    print("END simulation")
    #print("Stats:")
    print(stats)



