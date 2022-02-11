import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import math
import simpy as simpy

# Collision detection type
# 0 - simple
# 1 - advanced
# 2 - full
collision_detection_type = 0

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

# global
nodes = []
packetsAtBS = []
joinRequestAtBS = []
env = simpy.Environment()

coding_rate = 1

nrCollisions = 0
nrReceived = 0
nrProcessed = 0
nrLost = 0
nrSent = 0
nrRetransmission = 0
nr_join_req_sent = 0
nr_join_req_dropped = 0
nr_join_acp_sent = 0
nr_sack_sent = 0

Ptx = 14
gamma = 2.08
d0 = 40.0
var = 0
Lpld0 = 127.41
GL = 0
power_threshold = 6
npream = 8
max_packets = 500
full_collision = True
retrans_count = 8

# max distance between nodes and base station (currently constant, later will be calculated)
max_dist = 100

# base station position
bsx = max_dist + 10
bsy = max_dist + 10
x_max = bsx + max_dist + 10
y_max = bsy + max_dist + 10

req_pack_len = 20
send_req = set()
send_sack = set()

min_nr_slots = 10
nr_connected_nodes = 0
slots = np.zeros(min_nr_slots)

# prepare graphics and draw base station
if graphics:
	plt.xlim([0, x_max])
	plt.ylim([0, y_max])
	ax.add_artist(plt.Circle((bsx, bsy), 3, fill=True, color='green'))
	ax.add_artist(plt.Circle((bsx, bsy), max_dist, fill=False, color='green'))


def add_nodes(node_count):
	global nodes
	print()
	print("Node initialization:")
	for i in range(len(nodes), node_count + len(nodes)):
		nodes.append(node(i, avg_wake_up_time, data_size))
	print()


# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf
def airtime(sf, cr, pl, bw):
	H = 0  # implicit header disabled (H=0) or not (H=1)
	DE = 0  # low data rate optimization enabled (=1) or not (=0)
	Npream = 8  # number of preamble symbol (12.25  from Utz paper)

	if bw == 125 and sf in [11, 12]:
		# low data rate optimization mandated for BW125 with SF11 and SF12
		DE = 1
	if sf == 6:
		# can only have implicit header with SF6
		H = 1

	Tsym = (2.0 ** sf) / bw
	Tpream = (Npream + 4.25) * Tsym
	# log(env, "sf", sf, " cr", cr, "pl", pl, "bw", bw)
	payloadSymbNB = 8 + max(math.ceil((8.0 * pl - 4.0 * sf + 28 + 16 - 20 * H) / (4.0 * (sf - 2 * DE))) * (cr + 4), 0)
	Tpayload = payloadSymbNB * Tsym
	return Tpream + Tpayload


class packet():
	def __init__(self, nodeid, pl, distance):
		global Ptx
		global gamma
		global d0
		global var
		global Lpld0
		global GL

		self.nodeid = nodeid
		self.txpow = Ptx

		self.cr = coding_rate
		self.sf = 7
		self.bw = 125

		Lpl = Lpld0 + 10 * gamma * math.log10(distance / d0)
		Prx = self.txpow - GL - Lpl

		# log-shadow
		#log(env, "Lpl:", Lpl)

		# transmission range, needs update XXX
		self.transRange = 150
		self.pl = pl
		self.symTime = (2.0 ** self.sf) / self.bw
		self.rssi = Prx
		self.freq = 860000000

		#log(env, "frequency", self.freq, "symTime ", self.symTime)
		#log(env, "bw", self.bw, "sf", self.sf, "cr", self.cr, "rssi", self.rssi)
		self.rectime = airtime(self.sf, self.cr, self.pl, self.bw)
		#log(env, "rectime node ", self.nodeid, "  ", self.rectime)

		# denote if packet is collided
		self.collided = 0
		self.processed = 0
		self.lost = False
		self.is_sack = False


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
	if abs(p1.rssi - p2.rssi) < power_threshold:
		# packets are too close to each other, both collide
		# return both packets as casualties
		return p1, p2
	elif p1.rssi - p2.rssi < power_threshold:
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
	p2_end = p2.addTime + p2.rectime
	p1_cs = env.now + Tpreamb
	if p1_cs < p2_end:
		# p1 collided with p2 and lost
		return True
	return False


# TODO: To check how to keep track of several collisions for a single packet (or how was it done before???)
def check_collision(packet):
	processing = 0
	for i in range(0, len(packetsAtBS)):
		if packetsAtBS[i].packet.processed == 1:
			processing = processing + 1
	if processing > max_packets:
		log(env, "too long:", len(packetsAtBS))
		packet.processed = 0
	else:
		packet.processed = 1

	if len(send_sack) != 0 and not packet.is_sack:
		log(env, "packet is dropped")
		packet.processed = 0

	if packetsAtBS:
		#log(env, "CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(
			#packet.nodeid, packet.sf, packet.bw, packet.freq,
			#len(packetsAtBS)))
		for other in packetsAtBS:
			if other.nodeid != packet.nodeid:
				#log(env, ">> node {} (sf:{} bw:{} freq:{:.6e})".format(
				#	other.nodeid, other.packet.sf, other.packet.bw, other.packet.freq))
				# simple collision
				if frequency_collision(packet, other.packet) and sf_collision(packet, other.packet):
					if full_collision:
						if timing_collision(packet, other.packet):
							# check who collides in the power domain
							c = power_collision(packet, other.packet)
							# mark all the collided packets
							# either this one, the other one, or both
							for p in c:
								p.collided = 1
								nrCollisions += 1
								log("COLLISION! node {} collided with node {}".format(p.nodeid, packet.nodeid))
						else:
							# no timing collision, all fine
							pass
					else:
						packet.collided = 1
						other.packet.collided = 1  # other also got lost, if it wasn't lost already
						nrCollisions += 1
						log("COLLISION! node {} collided with node {}".format(packet.nodeid, other.packet.nodeid))


class gateway():
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.join_acp_arr = []

	def join_acp(self, node, env):
		global send_req
		global req_pack_len
		global nr_connected_nodes
		global nrCollisions
		acp_packet = packet(-1, req_pack_len, node.dist)
		acp_packet.sf = 12
		acp_packet.bw = 125
		acp_packet.freq = 864000000
		yield env.timeout(1000)

		if len(self.join_acp_arr) == 0:
			self.join_acp_arr.append(acp_packet)
		else:
			yield env.timeout(self.join_acp_arr[0].rectime)
			self.join_acp_arr = []
			log(env, "gateway dropped join req from node {}".format(node.nodeid))
			global nr_join_req_dropped
			nr_join_req_dropped += 1
			return

		global nr_join_acp_sent
		nr_join_acp_sent += 1
		# log(env, "gateway sent join accept to node {} \tSF:{}\tdata size:{}b\trssi:{:.3f}dBm\tfreq:{:.1f}MHZ\tbw:{}kHz\tairtime:{:.3f}s".format(
		# 	node.nodeid, acp_packet.sf, acp_packet.pl, acp_packet.rssi, acp_packet.freq / 1000000.0,
		# 	acp_packet.bw, acp_packet.rectime/1000))

		for n in send_req:
			if n == node:    continue

			if timing_collision(acp_packet, n.req_packet):
				log(env, 'join req failed: collision at the gateway')
				n.req_packet.collided = True
				nrCollisions += 1

		# assume that all nodes send request in SF 12, so no need to check for imperfect orthogonality
		for n in send_req:
			if power_collision(acp_packet, n.req_packet) and \
			frequency_collision(acp_packet, n.req_packet) and \
			timing_collision(acp_packet, n.req_packet):
				log(env, 'join accept failed: collision at the node')
				nrCollisions += 1
				return

		node.accept_received = True

join_gateway = gateway(0, 0)


class node():
	def __init__(self, nodeid, period, packetlen):
		global nr_connected_nodes

		self.nodeid = nodeid
		self.period = period
		# self.bs = bs
		self.x = 0
		self.y = 0
		self.retry_count = 0

		self.round_start_time = 0
		self.round_end_time = 0
		self.missed_sack_packet = 0
		self.connected = False
		self.accept_received = False
		self.waiting_first_sack = False
		self.slot = -1
		self.guard_time = 0
		self.network_size = 0
		self.sack_packet_received = env.event()

		global nodes
		global Ptx
		self.txpow = Ptx
		found = False
		rounds = 0

		while not found and rounds < 100:
			a = random.random()
			b = random.random()
			if b < a:
				a, b = b, a
			posx = b * max_dist * math.cos(2 * math.pi * a / b) + bsx
			posy = b * max_dist * math.sin(2 * math.pi * a / b) + bsy

			if len(nodes) > 0:
				for index, n in enumerate(nodes):
					dist = np.sqrt(((abs(n.x - posx)) ** 2) + ((abs(n.y - posy)) ** 2))
					if dist >= 10:
						found = True
						self.x = posx
						self.y = posy
					else:
						rounds += 1
						if rounds == 100:
							log(env, "could not place new node, giving up")
							exit(-1)
			else:
				#print("first node")
				self.x = posx
				self.y = posy
				found = True
		self.dist = np.sqrt((self.x - bsx) * (self.x - bsx) + (self.y - bsy) * (self.y - bsy))
		print("node {}: \tx {:.3f}\ty {:.3f}\tdist:{:.3f}".format(nodeid, self.x, self.y, self.dist))

		self.packet = packet(self.nodeid, packetlen, self.dist)
		self.sent = 0

		global graphics
		if graphics:
			global ax
			ax.add_artist(plt.Circle((self.x, self.y), 2, color='blue'))

	def init_req_packet(self):
		self.req_packet = packet(self.nodeid, req_pack_len, self.dist)
		self.req_packet.sf = 12
		self.req_packet.bw = 125
		self.req_packet.freq = 864000000
		self.req_packet.addTime = env.now

	def join_req(self, env):
		global nodes
		global d0
		global req_pack_len
		global send_req
		global join_gateway
		global nrRetransmission
		global nr_join_req_sent

		self.init_req_packet()
		# to count how many times we resend a req_packet

		# all the nodes that have already sent the request, but haven't got
		# accept from gateway yet
		# needed to check collisions
		send_req.add(self)
		# if distance between node and gateway is ok, so path loss is not
		# too big
		dist = np.sqrt(self.x * self.x + self.y * self.y)

		# update statistics count
		nr_join_req_sent += 1
		log(env, "node {} sent join request \t\t\t\tSF:{}\tdata size:{}b\trssi:{:.3f}dBm\tfreq:{:.1f}MHZ\tbw:{}kHz\tairtime:{:.3f}s".format(
			self.nodeid, self.req_packet.sf, self.req_packet.pl, self.req_packet.rssi, self.req_packet.freq / 1000000.0,
			self.req_packet.bw, self.req_packet.rectime/1000))

		pl = Lpld0 + 10 * gamma * math.log10(dist / d0)
		Prx = self.txpow - GL - pl
		if Prx < -133.25:
			log(env, "node {} join request failed, too much path loss: {}".format(self.nodeid, Prx))
			yield env.timeout(self.req_packet.rectime)
			send_req.remove(self)
			return

		# check collision with other req packets
		def check_req_coll():
			log(env, "CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(
				self.req_packet.nodeid, self.req_packet.sf, self.req_packet.bw, self.req_packet.freq,
				len(send_req)))
			for node in send_req:
				log(env, "CHECK >> node {} (sf:{} bw:{} freq:{:.6e})".format(
					node.req_packet.nodeid, node.req_packet.sf, node.req_packet.bw, node.req_packet.freq))
				if node == self:
					continue

				if timing_collision(self.req_packet, node.req_packet) and \
					power_collision(self.req_packet, node.req_packet) and \
						frequency_collision(self.req_packet, node.req_packet):
					log(env, 'join request failed, retrying')
					return False
			return True

		while True:
			if check_req_coll():
				# check if accept received
				yield env.process(join_gateway.join_acp(self, env))
				if not self.accept_received:
					yield env.timeout(self.req_packet.rectime) 	# to add some randomness to waiting time before retransmission
					yield env.timeout(1000)  					# waiting time = 4.5 s + random between 0 and 5 s
					self.init_req_packet()
				else:
					yield env.timeout(self.req_packet.rectime)
					send_req.remove(self)
					self.retry_count = 0
					self.connected = True

					# to deliver this information in SACK packet, not in join accept packet
					reserve_slot(self)
					self.network_size = len(slots)
					self.waiting_first_sack = True
					return

			else:
				log(env, "COLLISION")
				yield env.timeout(self.req_packet.rectime)
				yield env.timeout(1000)
				self.init_req_packet()

			self.retry_count += 1
			nrRetransmission += 1
			log(env, "node {} sent join request RETRANSMISSION (retry count = {})".format(self.nodeid, self.retry_count))
			# yield env.timeout(self.req_packet.rectime)
			if self.retry_count >= retrans_count:
				send_req.remove(self)
				log(env, "Request failed, too many retries {}".format(self.retry_count))
				return

			if self.req_packet.collided:
				yield env.timeout(1000)
				self.init_req_packet()

#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#
def transmit(env, node):
	while True:
		if not node.connected and node.retry_count < retrans_count:
			yield env.timeout(random.expovariate(1.0 / float(node.period)))  # wake up at random time
			yield env.process(node.join_req(env))
			if node.connected:
				log(env, "node {} connected".format(node.nodeid))
			else:
				log(env, "node {} connection failed".format(node.nodeid))
		if not node.connected:	break

		if node.waiting_first_sack:
			yield node.sack_packet_received # timeout = 3 default size of frame => then do join request again
			node.waiting_first_sack = False
			node.sack_packet_received = env.event()
		else:
			yield env.timeout(node.round_end_time - env.now)

		if node.round_start_time < env.now:
			log(env, "node {}: missed sack packet".format(node.nodeid))
			node.round_start_time = env.now + 1
			node.missed_sack_packet += 1
		else:
			node.missed_sack_packet = 0

		if node.missed_sack_packet == 3:
			log(env, "node {}: reconnecting to the gateway. ".format(node.nodeid))
			node.connected = False
			discard_slot(node)
			continue

		yield env.timeout(node.round_start_time - env.now)
		node.round_end_time = env.now + frame_length(node.network_size)
		send_time = node.slot * (node.packet.rectime + 2 * node.guard_time) + node.guard_time
		yield env.timeout(send_time)

		# time sending and receiving
		# packet arrives -> add to base station

		global nrSent
		if node in packetsAtBS:
			log(env, "ERROR: packet already in")
		else:
			sensitivity = sensitivities[node.packet.sf - 7, [125, 250, 500].index(node.packet.bw) + 1]
			if node.packet.rssi < sensitivity:
				log(env, "node {}: packet will be lost".format(node.nodeid))
				node.packet.lost = True
			else:
				node.packet.lost = False
				check_collision(node.packet)
				packetsAtBS.append(node)
				node.packet.addTime = env.now

		node.sent = node.sent + 1
		nrSent += 1
		# log(env, "node {} sent data packet\t\t\t\tSF:{}\tdata size:{}b\trssi:{:.3f}dBm\tfreq:{:.1f}MHZ\tbw:{}kHz\tairtime:{:.3f}s\tguardtime:{:.3f}ms".format(
		# 	node.nodeid, node.packet.sf, node.packet.pl, node.packet.rssi, node.packet.freq/1000000.0, node.packet.bw, node.packet.rectime/1000, node.guard_time))
		yield env.timeout(node.packet.rectime)
		update_statistics(node.packet)

		# complete packet has been received by base station
		# can remove it
		if node in packetsAtBS:
			packetsAtBS.remove(node)
		reset_packet(node.packet)


def update_statistics(packet):
	if packet.lost:
		global nrLost
		nrLost += 1
	if packet.collided == 1:
		global nrCollisions
		nrCollisions = nrCollisions + 1
	if packet.collided == 0 and not packet.lost:
		global nrReceived
		nrReceived = nrReceived + 1
	if packet.processed == 1:
		global nrProcessed
		nrProcessed = nrProcessed + 1


def reset_packet(packet):
	packet.collided = 0
	packet.processed = 0
	packet.lost = False


def is_packet_delivered(packet):
	return packet.collided == 0 and \
		   packet.processed == 1 and \
		   not packet.lost


def transmit_sack_to_node(env, node, sack_packet_len, guard_time, nr_slots):
	sack_packet = packet(-1, sack_packet_len, node.dist)
	sack_packet.is_sack = True

	check_collision(sack_packet)
	send_sack.add(sack_packet)

	yield env.timeout(sack_packet.rectime)

	update_statistics(sack_packet)
	send_sack.remove(sack_packet)

	if is_packet_delivered(sack_packet):
		node.round_start_time = env.now + guard_time + 1
		node.network_size = nr_slots
		node.guard_time = guard_time
		if node.waiting_first_sack:
			node.sack_packet_received.succeed()

	reset_packet(sack_packet)


def transmit_sack(env):
	# calculating time when the first sack packet will be sent
	sack_packet_len = 255
	frame_len = frame_length(min_nr_slots)
	guard_time = frame_len * 3 * 0.0001
	sack_slot_len = airtime(7, coding_rate, sack_packet_len, 125) + 2 * guard_time

	transmission_time = random.uniform(0, frame_len - sack_slot_len)

	# main sack packet transmission loop
	while True:
		yield env.timeout(transmission_time)

		nr_slots = len(slots)
		sack_packet_len = 255
		frame_len = frame_length(nr_slots)
		guard_time = frame_len * 3 * 0.0001

		prev_sack_slot_len = sack_slot_len
		sack_packet_rectime = airtime(7, coding_rate, sack_packet_len, 125)
		sack_slot_len = sack_packet_rectime + 2 * guard_time
		round_start_time = env.now + sack_packet_rectime + guard_time + 1

		transmission_time = prev_sack_slot_len + 1 + frame_len - sack_slot_len

		global nr_sack_sent
		nr_sack_sent += 1
		log(env, "gateway sent SACK packet to the nodes")

		for n in nodes:
			if n.connected:
				env.process(transmit_sack_to_node(env, n, sack_packet_len, guard_time, nr_slots))


def frame_length(nr_slots):
	sack_packet_len = 255
	sack_packet_airtime = airtime(7, coding_rate, sack_packet_len, 125)
	data_packet_airtime = airtime(7, coding_rate, data_size, 125)
	return (nr_slots * data_packet_airtime + sack_packet_airtime) / (1 - (6 * nr_slots + 1) * 0.0001)


def reserve_slot(node):
	global slots
	global nr_connected_nodes
	if nr_connected_nodes < len(slots):
		slot_index = np.where(slots == 0)[0][0]
		slots[slot_index] = node.nodeid + 1
		node.slot = slot_index
	else:
		slots = np.append(slots, node.nodeid + 1)
		node.slot = len(slots) - 1
	nr_connected_nodes += 1
	return


def discard_slot(node):
	global slots
	global nr_connected_nodes
	if node.slot == -1:
		return
	slots[node.slot] = 0
	node.slot = -1
	nr_connected_nodes -= 1
	return


def log(env, str):
	print("{:.3f} s: {}".format(env.now / 1000, str))


def start_simulation():
	for n in nodes:
		env.process(transmit(env, n))
	env.process(transmit_sack(env))
	print("Simulation start")
	env.run(until=sim_time)
	print("Simulation finished\n")

def show_final_statistics():
	print("Collisions:", nrCollisions)
	print("Lost packets:", nrLost)
	print("Transmitted data packets:", nrSent)
	for n in nodes:
		print("\tNode", n.nodeid, "sent", n.sent, "packets")
	print("Transmitted SACK packets:", nr_sack_sent)
	print("Transmitted join request packets:", nr_join_req_sent)
	print("Transmitted join accept packets:", nr_join_acp_sent)
	print("Retransmissions:", nrRetransmission)
	print("Join request packets dropped by gateway:", nr_join_req_dropped)
	# to add average join time
	# to add power consumption


if __name__ == '__main__':
	# get arguments
	if len(sys.argv) >= 1:
		nodes_count = int(sys.argv[1])
		data_size = int(sys.argv[2])
		avg_wake_up_time = int(sys.argv[3])
		collision_type = int(sys.argv[4])
		sim_time = (int(sys.argv[5]))

		print("Nodes:", nodes_count)
		print("Data size:", data_size, "bytes")
		print("Average wake up time of nodes (exp. distributed):", avg_wake_up_time, "seconds")
		print("Collision detection type:", collision_type)
		print("Simulation time:", sim_time, "seconds")

		avg_wake_up_time *= 1000
		sim_time *= 1000

		add_nodes(nodes_count)
		if graphics:
			plt.draw()
			plt.show(block=True)
	else:
		print("usage: ./main <number_of_nodes> <data_size> <avg_wake_up_time> <collision_type> <sim_time>")
		exit(-1)

start_simulation()
show_final_statistics()