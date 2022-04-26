import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import math
import simpy as simpy

graphics = True
fig, ax = plt.subplots()

# Arrays of measured sensitivity values
sf7 = np.array([7, -123, -120, -117.0])
sf8 = np.array([8, -126, -123, -120.0])
sf9 = np.array([9, -129, -126, -123.0])
sf10 = np.array([10, -132, -129, -126.0])
sf11 = np.array([11, -134.53, -131.52, -128.51])
sf12 = np.array([12, -137, -134, -131.0])
sensitivities = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

# IsoThresholds for collision detection caused by imperfect orthogonality of SFs
IS7 = np.array([1, -8, -9, -9, -9, -9])
IS8 = np.array([-11, 1, -11, -12, -13, -13])
IS9 = np.array([-15, -13, 1, -13, -14, -15])
IS10 = np.array([-19, -18, -17, 1, -17, -18])
IS11 = np.array([-22, -22, -21, -20, 1, -20])
IS12 = np.array([-25, -25, -25, -24, -23, 1])
iso_thresholds = np.array([IS7, IS8, IS9, IS10, IS11, IS12])

# power consumptions for transmitting, receiving, and operating in mA
pow_cons = [75, 45, 30]
V = 3.3  # voltage XXX

# global
join_gateway = None
data_gateway = None
nodes = []
packets_at_bs = []
join_request_at_bs = []
env = simpy.Environment()

coding_rate = 1

# Statistics
nr_collisions = 0
nr_received = 0
nr_processed = 0
nr_lost = 0
nr_packets_sent = 0
nr_data_packets_sent = 0
nrRetransmission = 0
nr_join_req_sent = 0
nr_join_req_dropped = 0
nr_join_acp_sent = 0
nr_sack_sent = 0

nr_joins = 0
total_join_time = 0
total_energy = 0

Ptx = 14
gamma = 2.08
d0 = 40.0
var = 0
Lpld0 = 127.41
GL = 0
power_threshold = 6
npream = 8
max_packets = 500
retrans_count = 8

sigma = 0.38795

# min waiting time before retransmission in s
min_wait_time = 4.5

# max distance between nodes and base station
max_dist = 400

# base station position
bsx = max_dist + 10
bsy = max_dist + 10
x_max = bsx + max_dist + 10
y_max = bsy + max_dist + 10

send_req = set()
send_sack = set()

# prepare graphics and draw base station
if graphics:
	plt.xlim([0, x_max])
	plt.ylim([0, y_max])
	ax.add_artist(plt.Circle((bsx, bsy), 3, fill=True, color='green'))
	ax.add_artist(plt.Circle((bsx, bsy), max_dist, fill=False, color='green'))


def add_nodes(node_count):
	global nodes
	global join_gateway
	print()
	print("Node initialization:")
	for i in range(len(nodes), node_count + len(nodes)):
		nodes.append(EndNode(i))
	print()


def frequency_collision(p1, p2):
	if abs(p1.freq - p2.freq) <= 120 and (p1.bw == 500 or p2.freq == 500):
		return True
	elif abs(p1.freq - p2.freq) <= 60 and (p1.bw == 250 or p2.freq == 250):
		return True
	else:
		if abs(p1.freq - p2.freq) <= 30:
			return True
	return False


def sf_collision(p1, p2):
	return p1.sf == p2.sf


def power_collision(p1, p2):  #
	p1_rssi, p2_rssi = p1.rssi(p2.node), p2.rssi(p1.node)
	if abs(p1_rssi - p2_rssi) < power_threshold:
		# packets are too close to each other, both collide
		# return both packets as casualties
		return p1, p2
	elif p1_rssi - p2_rssi < power_threshold:
		# p2 overpowered p1, return p1 as casualty
		return p1,
	# p2 was the weaker packet, return it as a casualty
	return p2,


def timing_collision(p1, p2):
	# assuming p1 is the freshly arrived packet and this is the last check
	# we've already determined that p1 is a weak packet, so the only
	# way we can win is by being late enough (only the first n - 5 preamble symbols overlap)

	# assuming 8 preamble symbols

	# we can lose at most (Npream - 5) * Tsym of our preamble
	Tpreamb = 2 ** p1.sf / (1.0 * p1.bw) * (npream - 5)

	# check whether p2 ends in p1's critical section
	p2_end = p2.add_time + p2.rec_time
	p1_cs = env.now + Tpreamb
	if p1_cs < p2_end:
		# p1 collided with p2 and lost
		return True
	return False


def get_sensitivity(sf, bw):
	return sensitivities[sf - 7, [125, 250, 500].index(bw) + 1]


# EU spectrum 863 - 870 MHz, LoraWAN IoT
# 868000000, 868200000, 868400000 - join request, join accept
# 867000000, 867200000, 867400000, 867600000, 867800000, 869525000(10%) - SF
# https://www.thethingsnetwork.org/docs/lorawan/frequency-plans/
class Channels:
	# Channels
	Channel = {
		1: 867000000,  # SF 7
		2: 867200000,  # SF 8
		3: 867400000,  # SF 9
		4: 867600000,  # SF 10
		5: 867800000,  # SF 11
		6: 868400000,  # SF 12
		7: 868000000,  # join request/accept
	}

	@classmethod
	def get_sf_freq(cls, sf):
		return cls.Channel[sf-6]

	@classmethod
	def get_jr_freq(cls):
		return cls.Channel[7]


class NetworkNode:
	def __init__(self, node_id=None):
		if node_id is not None:
			self.node_id = node_id
		self.x, self.y = 0, 0


class Gateway(NetworkNode):
	def __init__(self, node_id=None):
		super().__init__(node_id)
		self.x, self.y = bsx, bsy
	
	@staticmethod		
	def is_gateway():
		return True

	def __str__(self):
		return "gateway"


class JoinGateway(Gateway):
	def __init__(self, node_id):
		super().__init__(node_id)

	def join_acp(self, join_req, env):
		global send_req
		yield env.timeout(1000)
		acp_packet = JoinAccept(self, join_req.node)
		acp_packet.add_time = env.now

		if BroadcastTraffic.nr_join_acp > 0:
			yield BroadcastTraffic.add_and_wait(env, acp_packet)
			join_req.processed = False
			log(env, f"gateway dropped join req from node {join_req.node.node_id}")
			return
		
		log(env,
			f'{f"gateway sent join accept to node {join_req.node.node_id}":<40}'
			f'{f"SF: {acp_packet.sf} ":<10}'
			f'{f"Data size: {acp_packet.pl} b ":<20}'
			f'{f"RSSI: {acp_packet.rssi(join_req.node):.3f} dBm ":<25}'
			f'{f"Freq: {acp_packet.freq / 1000000.0:.3f} MHZ ":<24}'
			f'{f"BW: {acp_packet.bw}  kHz ":<18}'
			f'{f"Airtime: {acp_packet.rec_time / 1000.0:.3f} s ":<22}')

		acp_packet.check_collision()
		yield BroadcastTraffic.add_and_wait(env, acp_packet)
		acp_packet.update_statistics()
		if acp_packet.is_received():
			join_req.node.accept_received = True
			global nr_joins
			global total_join_time
			nr_joins += 1
			total_join_time += env.now

		acp_packet.reset()


class DataGateway(Gateway):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.frames = [Frame(sf) for sf in range(7,13)]

	def frame(self, sf):
		if sf > 6:
			return self.frames[sf-7]
		raise ValueError

	def transmit_sack(self, env, sf):
		# main sack packet transmission loop
		while True:
			yield env.timeout(self.frame(sf).trans_time - env.now)
			sack_packet = SackPacket(self.frame(sf).nr_slots_SACK, sf, self)

			# print("-" * 70)
			if self.frame(sf).nr_taken_slots != 0:
				log(env,
					f'{f"gateway sent SACK packet to the nodes: ":<40}'
					f'{f"SF: {sf} ":<10}'
					f'{f"Data size: {sack_packet.pl} b ":<20}'
					f'{"":<25}'
					f'{f"Freq: {sack_packet.freq / 1000000.0:.3f} MHZ ":<24}'
					f'{f"BW: {sack_packet.bw}  kHz ":<18}'
					f'{f"Airtime: {sack_packet.rec_time / 1000.0:.3f} s ":<22}')
			# print("-" * 70)

			for n in nodes:
				if n.connected and n.sf == sf:
					env.process(self.transmit_sack_to_node(env, n, sf))

			yield env.timeout(self.frame(sf).next_round_start_time - env.now)
			self.frame(sf).next_frame()

	def transmit_sack_to_node(self, env, node, sf):
		sack_packet = SackPacket(self.frame(sf).nr_slots_SACK, sf, self)
		sack_packet.add_time = env.now

		if sack_packet.is_lost(node):
			sack_packet.lost = True
			log(env, f"{self} transmit to {node} SACK failed, too much path loss: {sack_packet.rssi(node)}")

		sack_packet.check_collision()

		yield BroadcastTraffic.add_and_wait(env, sack_packet)
		sack_packet.update_statistics()

		if sack_packet.is_received():
			node.round_start_time = self.frame(sf).next_round_start_time
			node.network_size = self.frame(sf).nr_slots
			node.guard_time = self.frame(sf).guard_time
			node.frame_length = self.frame(sf).frame_length
			if node.waiting_first_sack:
				node.sack_packet_received.succeed()
		else:
			log(env, f"Sack packet was not received by {node}")
		sack_packet.reset()


class EndNode(NetworkNode):
	def __init__(self, node_id):
		super().__init__(node_id)

		self.join_retry_count = 0
		self.missed_sack_count = 0
		self.packets_sent_count = 0

		self.connected = False
		self.accept_received = False
		self.waiting_first_sack = False

		self.round_start_time = 0
		self.round_end_time = 0

		self.slot = None
		self.guard_time = 0
		self.frame_length = 0
		self.network_size = 0

		self.req_packet, self.data_packet = None, None
		self.sack_packet_received = env.event()

		self.x, self.y = EndNode.find_place_for_new_node()
		self.dist = np.sqrt((self.x - bsx) * (self.x - bsx) + (self.y - bsy) * (self.y - bsy))

		self.sf = self.find_optimal_sf()

		print(f"node {self.node_id}: \t x {self.x:3f} \t y {self.y:3f} \t dist {self.dist:4.3f} \t SF {self.sf}")

		self.draw()

	def __str__(self):
		return f"node {self.node_id}"

	@staticmethod
	def is_gateway():
		return False

	@staticmethod
	def find_place_for_new_node():
		global nodes
		found = False
		rounds = 0
		while not found and rounds < 100:
			a = random.random()
			b = random.random()
			if b < a:
				a, b = b, a
			posx = b * max_dist * math.cos(2 * math.pi * a / b) + bsx
			posy = b * max_dist * math.sin(2 * math.pi * a / b) + bsy

			if len(nodes) == 0:
				found = True
				break

			for index, n in enumerate(nodes):
				dist = np.sqrt(((abs(n.x - posx)) ** 2) + ((abs(n.y - posy)) ** 2))
				if dist >= 10:
					found = True
				else:
					found = False
					rounds += 1
					if rounds == 100:
						print("could not find place for a new node, giving up")
						exit(-1)

		return posx, posy

	def draw(self):
		global graphics
		if graphics:
			global ax
			ax.add_artist(plt.Circle((self.x, self.y), 2, color='blue'))

	def find_optimal_sf(self):
		for sf in range(7, 13):
			data_packet = DataPacket(sf, self)
			if data_packet.rssi(data_gateway) > get_sensitivity(sf, data_packet.bw):
				return sf
		print(f"WARNING: {self} cannot reach gateway!")
		return None

	def join_req(self, env):
		global nodes
		global d0
		global req_pack_len
		global join_gateway
		global nrRetransmission
		global nr_join_req_sent

		req_packet = JoinRequest(self)
		req_packet.add_time = env.now

		log(env,
			f'{f"node {self.node_id} sent join request ":<40}'
			f'{f"SF: {req_packet.sf} ":<10}'
			f'{f"Data size: {req_packet.pl} b ":<20}'
			f'{f"RSSI: {req_packet.rssi(join_gateway):.3f} dBm ":<25}'
			f'{f"Freq: {req_packet.freq / 1000000.0:.3f} MHZ ":<24}'
			f'{f"BW: {req_packet.bw}  kHz ":<18}'
			f'{f"Airtime: {req_packet.rec_time / 1000.0:.3f} s ":<22}')

		if req_packet.is_lost(join_gateway):
			req_packet.lost = True
			log(env, f"node {self.node_id} join request failed, too much path loss: {req_packet.rssi(join_gateway)}")

		while True:
			req_packet.check_collision()
			yield BroadcastTraffic.add_and_wait(env, req_packet)
			if req_packet.is_received():
				# check if accept received
				yield env.process(join_gateway.join_acp(req_packet, env))
				if self.accept_received:
					self.join_retry_count = 0
					self.connected = True

					req_packet.update_statistics()
					data_gateway.frame(self.sf).add(self)
					self.waiting_first_sack = True
					return

			req_packet.update_statistics()

			# waiting time before retransmission = 4.5 s + random between 0 and 5 s
			yield env.timeout(min_wait_time + (random.uniform(0.0, 5.0)) * 1000)
			req_packet = JoinRequest(self)
			req_packet.add_time = env.now
			self.join_retry_count += 1
			nrRetransmission += 1

			log(env, f"{self} sent join request RETRANSMISSION (retry count = {self.join_retry_count})")
			if self.join_retry_count >= retrans_count:
				log(env, f"Request failed, too many retries: {self.join_retry_count}")
				return

	# main discrete event loop, runs for each node
	def transmit(self, env):
		while True:
			# connecting to the gateway
			if not self.connected and self.join_retry_count < retrans_count:
				yield env.timeout(random.expovariate(1.0 / float(avg_wake_up_time)))  # wake up at random time
				yield env.process(self.join_req(env))
				if not self.connected:
					log(env, f"node {self.node_id} connection failed")
					continue
				log(env, f"node {self.node_id} connected")

			# giving up in case of too many retransmissions
			if not self.connected:
				break

			# waiting for sack packet
			if self.waiting_first_sack:
				yield self.sack_packet_received  # timeout = 3 default size of frame => then do join request again
				self.waiting_first_sack = False
				self.sack_packet_received = env.event()
			else:
				yield env.timeout(self.round_end_time - env.now)

			# calculating round start time
			if self.round_start_time < env.now:
				log(env, f"{self}: missed sack packet")
				self.round_start_time = env.now + 1
				self.missed_sack_count += 1
			else:
				self.missed_sack_count = 0

			# reconnecting to gateway if too many SACK-s missed
			if self.missed_sack_count == 3:
				log(env, "node {}: reconnecting to the gateway. ".format(self.node_id))
				self.connected = False
				data_gateway.frame(self.sf).remove(self)
				continue

			# waiting till round starts
			yield env.timeout(self.round_start_time - env.now)

			# calculating round_end_time and waiting till send_time
			self.round_end_time = env.now + self.frame_length
			send_time = self.slot * (DataPacket(self.sf).rec_time + 2 * self.guard_time) + self.guard_time
			if self.slot != 0:
				send_time = send_time + random.gauss(0, sigma) * self.guard_time
			yield env.timeout(send_time)

			# time sending and receiving
			# packet arrives -> add to base station

			data_packet = DataPacket(self.sf, self)
			data_packet.add_time = env.now
			data_packet.sent = True

			log(env,
				f'{f"node {self.node_id} sent data packet ":<40}'
				f'{f"SF: {data_packet.sf} ":<10}'
				f'{f"Data size: {data_packet.pl} b ":<20}'
				f'{f"RSSI: {data_packet.rssi(data_gateway):.3f} dBm ":<25}'
				f'{f"Freq: {data_packet.freq / 1000000.0:.3f} MHZ ":<24}'
				f'{f"BW: {data_packet.bw}  kHz ":<18}'
				f'{f"Airtime: {data_packet.rec_time / 1000.0:.3f} s ":<22}'
				f'{f"Guardtime: {self.guard_time / 1000.0:.3f} ms"}')

			if data_packet.rssi(data_gateway) < get_sensitivity(data_packet.sf, data_packet.bw):
				log(env, f"{self}: packet will be lost")
				data_packet.lost = True

			data_packet.check_collision()
			yield BroadcastTraffic.add_and_wait(env, data_packet)
			data_packet.update_statistics()
			data_packet.reset()


class Packet:
	def __init__(self, node=None, receiver=None):
		if node is None:
			self.node = NetworkNode()
		else:
			self.node = node

		if receiver is None:
			self.receiver = None
		else:
			self.receiver = receiver

		self.cr = coding_rate
		self.bw, self.sf, self.pl, self.rec_time = 0, 0, 0, 0

		self.collided = False
		self.processed = False
		self.lost = False
		self.sent = False
		self.add_time = None

	def energy_transmit(self):
		return self.airtime() * (pow_cons[0] + pow_cons[2]) * V / 1e6
	
	def energy_receive(self):
		if self.is_received():
			return (Frame(self.sf).guard_time + self.airtime()) * (pow_cons[1] + pow_cons[2]) * V / 1e6 
		return 0

	def dist(self, destination):
		return np.sqrt((self.node.x - destination.x) * (self.node.x - destination.x) + (self.node.y - destination.y) * (self.node.y - destination.y))

	# TODO: add variance, sigma so that 0.5% packets will be...
	def rssi(self, destination):
		Lpl = Lpld0 + 10 * gamma * math.log10(self.dist(destination) / d0)
		Prx = Ptx - GL - Lpl
		return Prx

	def is_lost(self, destination):
		rssi = self.rssi(destination)
		sens = get_sensitivity(self.sf, self.bw)
		return rssi < sens

	# this function computes the airtime of a packet according to LoraDesignGuide_STD.pdf
	def airtime(self):
		H = 0  # implicit header disabled (H=0) or not (H=1)
		DE = 0  # low data rate optimization enabled (=1) or not (=0)
		Npream = 8  # number of preamble symbol (12.25  from Utz paper)

		if self.bw == 125 and self.sf in [11, 12]:
			DE = 1		# low data rate optimization mandated for BW125 with SF11 and SF12
		if self.sf == 6:
			H = 1		# can only have implicit header with SF6

		Tsym = (2.0 ** self.sf) / self.bw
		Tpream = (Npream + 4.25) * Tsym
		payloadSymbNB = 8 + max(math.ceil((8.0 * self.pl - 4.0 * self.sf + 28 + 16 - 20 * H) / (4.0 * (self.sf - 2 * DE))) * (self.cr + 4), 0)
		Tpayload = payloadSymbNB * Tsym
		return Tpream + Tpayload

	def reset(self):
		self.collided = False
		self.processed = False
		self.lost = False
	
	def update_statistics(self):
		if self.lost:
			global nr_lost
			nr_lost += 1

		if self.collided:
			global nr_collisions
			nr_collisions += 1

		if self.is_received():
			global nr_received
			nr_received += 1

		if self.processed:
			global nr_processed
			nr_processed += 1

		if self.sent:
			global nr_packets_sent
			nr_packets_sent += 1

		global total_energy
		total_energy += (self.energy_transmit() + self.energy_receive())

	def is_received(self):
		return not self.collided and self.processed and not self.lost

	def was_sent_to(self, node):
		return self.receiver is node

	def check_collision(self):
		self.processed = True
		if BroadcastTraffic.nr_data_packets > max_packets:
			log(env, "too many packets are being sent to the gateway:", BroadcastTraffic)
			self.processed = False

		if BroadcastTraffic.nr_packets:
			for other in BroadcastTraffic.traffic:
				if self.node is other.node:
					continue

				if self.node.is_gateway() != other.node.is_gateway() and self.sf == other.sf:
					if self.processed and self.was_sent_to(other.node):
						log(env, f"{self} from {self.node} is dropped")
						self.processed = False

					if other.processed and other.was_sent_to(self.node):
						log(env, f"{other} from {other.node} is dropped")
						other.processed = False

				if frequency_collision(self, other) and \
					sf_collision(self, other) and \
					timing_collision(self, other):

					for p in power_collision(self, other):
						p.collided = True
						if p == self:
							p2 = other
						else:
							p2 = self
						log(env, f"COLLISION! {p.node} collided with {p2.node}")


class DataPacket(Packet):
	def __init__(self, sf=None, node=None):
		super().__init__(node, data_gateway)
		self.sf = sf
		self.bw = 125
		self.freq = Channels.get_sf_freq(sf)
		self.pl = data_size
		self.rec_time = self.airtime()
	
	def update_statistics(self):
		super().update_statistics()
		if self.sent:
			global nr_data_packets_sent
			nr_data_packets_sent += 1

		if self.sent and self.node is not None:
			self.node.packets_sent_count += 1

	def __str__(self):
		return "data packet"


class SackPacket(Packet):
	def __init__(self, nr_slots, sf=None, node=None):
		super().__init__(node, None)
		self.sf = sf
		self.bw = 125
		self.freq = Channels.get_sf_freq(sf)
		self.pl = int(4 + (nr_slots+7)/8)
		self.rec_time = self.airtime()
		
	def update_statistics(self):
		super().update_statistics()
		if self.sent:
			global nr_sack_sent
			nr_sack_sent += 1

	def was_sent_to(self, node):
		return self.sf == node.sf

	def __str__(self):
		return "SACK packet"


class JoinRequest(Packet):
	def __init__(self, node=None):
		super().__init__(node, join_gateway)
		self.sf = 12
		self.bw = 125
		self.freq = Channels.get_jr_freq()
		self.pl = 18
		self.rec_time = self.airtime()
		
	def update_statistics(self):
		super().update_statistics()

		if not self.processed:
			global nr_join_req_dropped
			nr_join_req_dropped += 1

		if self.sent:
			global nr_join_req_sent
			nr_join_req_sent += 1

	def __str__(self):
		return "join request"


class JoinAccept(Packet):
	def __init__(self, node, receiver):
		super().__init__(node, receiver)
		self.sf = 12
		self.bw = 125
		self.freq = Channels.get_jr_freq()
		self.pl = 12
		self.rec_time = self.airtime()
	
	def update_statistics(self):
		super().update_statistics()
		if self.sent:
			global nr_join_acp_sent
			nr_join_acp_sent += 1

	def __str__(self):
		return "join accept"


class Frame:
	def __init__(self, sf):
		self.sf = sf
		self.data_p_rec_time = DataPacket(sf).rec_time

		self.min_frame_length = 100 * self.data_p_rec_time
		self.guard_time = 3 * 0.0001 * self.min_frame_length
		self.min_nr_slots = int(self.min_frame_length/(self.data_p_rec_time + 2 * self.guard_time))

		self.nr_slots = self.min_nr_slots
		self.nr_taken_slots = 0
		self.nr_slots_SACK = self.nr_slots

		self.frame_length = self.min_frame_length

		self.sack_p_rec_time = SackPacket(self.nr_slots, sf).rec_time
		self.data_slot_len = self.data_p_rec_time + 2 * self.guard_time
		self.sack_slot_len = self.sack_p_rec_time + 2 * self.guard_time

		self.trans_time = random.uniform(0, self.frame_length - self.sack_slot_len)
		self.trans_time_period = self.sack_p_rec_time + self.guard_time
		self.next_round_start_time = self.trans_time + self.trans_time_period + 1

		self.slots = [None for _ in range(self.nr_slots)]

	def __update_fields(self):
		self.sack_p_rec_time = SackPacket(self.nr_slots, self.sf)
		if self.nr_taken_slots > self.min_nr_slots:
			self.frame_length = (self.nr_slots * self.data_p_rec_time + self.sack_p_rec_time)/(1.0 - 6 * 0.0001 * (self.nr_slots + 1))
		self.guard_time = 3 * 0.0001 * self.frame_length
		self.data_slot_len = self.data_p_rec_time + 2 * self.guard_time
		self.sack_slot_len = self.sack_p_rec_time + 2 * self.guard_time

	def next_frame(self):
		self.trans_time = self.next_round_start_time + self.frame_length - self.sack_slot_len + self.guard_time
		self.trans_time_period = self.sack_p_rec_time + self.guard_time
		self.next_round_start_time = self.trans_time + self.trans_time_period + 1
		self.nr_slots_SACK = self.nr_slots

	def add(self, node):
		if self.nr_taken_slots < self.nr_slots:
			slot = self.slots.index(None)
			node.slot = slot
			self.slots[slot] = node
		else:
			node.slot = self.nr_slots
			self.slots.append(node)
			self.nr_slots += 1
			self.__update_fields()
		self.nr_taken_slots += 1

	def remove(self, node):
		if node.slot is None:
			return
		self.slots[node.slot] = None
		node.slot = None
		self.nr_taken_slots -= 1


class BroadcastTraffic:
	traffic = []
	nr_packets = 0
	nr_data_packets = 0
	nr_sack_packets = 0
	nr_join_req = 0
	nr_join_acp = 0

	@classmethod
	def __inc_count(cls, packet):
		cls.nr_packets += 1
		if isinstance(packet, DataPacket):
			cls.nr_data_packets += 1
		if isinstance(packet, SackPacket):
			cls.nr_sack_packets += 1
		if isinstance(packet, JoinRequest):
			cls.nr_join_req += 1
		if isinstance(packet, JoinAccept):
			cls.nr_join_acp += 1

	@classmethod
	def __dec_count(cls, packet):
		cls.nr_packets -= 1
		if isinstance(packet, DataPacket):
			cls.nr_data_packets -= 1
		if isinstance(packet, SackPacket):
			cls.nr_sack_packets -= 1
		if isinstance(packet, JoinRequest):
			cls.nr_join_req -= 1
		if isinstance(packet, JoinAccept):
			cls.nr_join_acp -= 1

	@classmethod
	def add_generator(cls, env, packet):
		cls.traffic.append(packet)
		cls.__inc_count(packet)
		yield env.timeout(packet.rec_time)
		packet.sent = True
		cls.__dec_count(packet)
		cls.traffic.remove(packet)

	@classmethod
	def add_and_wait(cls, env, packet):
		return env.process(cls.add_generator(env, packet))

	@classmethod
	def is_p_cls_broadcasting(cls, packet_class):
		if packet_class == DataPacket:
			return cls.nr_data_packets > 0
		if packet_class == SackPacket:
			return cls.nr_sack_packets > 0
		if packet_class == JoinRequest:
			return cls.nr_join_req > 0
		if packet_class == JoinAccept:
			return cls.nr_join_acp > 0
		return False


def log(env, str):
	print(f'{f"{env.now/1000:.3f} s":<12} {str}')


def start_simulation():
	for n in nodes:
		env.process(n.transmit(env))
	for sf in range(7, 13):
		env.process(data_gateway.transmit_sack(env, sf))
	print("Simulation start")
	env.run(until=sim_time)
	print("Simulation finished\n")


def show_final_statistics():
	print("Collisions:", nr_collisions)
	print("Lost packets:", nr_lost)
	print("Transmitted data packets:", nr_data_packets_sent)
	for n in nodes:
		print("\tNode", n.node_id, "sent", n.packets_sent_count, "packets")
	print("Transmitted SACK packets:", nr_sack_sent)
	print("Transmitted join request packets:", nr_join_req_sent)
	print("Transmitted join accept packets:", nr_join_acp_sent)
	print("Retransmissions:", nrRetransmission)
	print("Join request packets dropped by gateway:", nr_join_req_dropped)
	print(f"Average join time: {total_join_time*0.001/nr_joins:.3f} s")
	print(f"Total energy consumption: {total_energy:.3f} J")
	print(f"Average energy consumption per node: {total_energy / nodes_count:.3f} J")
	print(f"Number of nodes failed to connect to the network:", nodes_count - nr_joins)


if __name__ == '__main__':
	# get arguments
	if len(sys.argv) >= 2:
		nodes_count = int(sys.argv[1])
		data_size = int(sys.argv[2])
		avg_wake_up_time = int(sys.argv[3])
		sim_time = (int(sys.argv[4]))

		print("Nodes:", nodes_count)
		print("Data size:", data_size, "bytes")
		print("Average wake up time of nodes (exp. distributed):", avg_wake_up_time, "seconds")
		print("Simulation time:", sim_time, "seconds")

		avg_wake_up_time *= 1000
		sim_time *= 1000

		join_gateway = JoinGateway(-1)
		data_gateway = DataGateway(-1)
		add_nodes(nodes_count)
		if graphics:
			plt.draw()
			plt.show(block=True)\

		start_simulation()
		show_final_statistics()
	else:
		print("usage: ./main <number_of_nodes> <data_size> <avg_wake_up_time> <sim_time>")
		exit(-1)
