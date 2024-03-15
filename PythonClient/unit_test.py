"""
Unit tests for Python client
"""

import unittest
from packets import *
from technology import *

class TestTechnology(unittest.TestCase):
    def testInitiallyResearchable(self):
        tree = TechnologyTree()
        researchable = tree.get_researchable()
        self.assertListEqual(researchable, [
            tree.techs['alphabet'],
            tree.techs['ceremonial_burial'],
            tree.techs['pottery'],
            tree.techs['masonry'],
            tree.techs['horseback_riding'],
            tree.techs['bronze_working'],
            tree.techs['warrior_code'],
        ])

    def testResearchByTechnologyObject(self):
        tree = TechnologyTree()
        tree.research(tree.get_researchable()[0])
        researchable = tree.get_researchable()
        self.assertListEqual(researchable, [
            tree.techs['ceremonial_burial'],
            tree.techs['pottery'],
            tree.techs['masonry'],
            tree.techs['horseback_riding'],
            tree.techs['bronze_working'],
            tree.techs['warrior_code'],
            tree.techs['writing'],
            tree.techs['code_of_laws'],
            tree.techs['map_making'],
        ])

    def testResearchByName(self):
        tree = TechnologyTree()
        tree.research('alphabet')
        tree.research('pottery')
        tree.research('map_making')
        researchable = tree.get_researchable()
        self.assertListEqual(researchable, [
            tree.techs['ceremonial_burial'],
            tree.techs['masonry'],
            tree.techs['horseback_riding'],
            tree.techs['bronze_working'],
            tree.techs['warrior_code'],
            tree.techs['writing'],
            tree.techs['code_of_laws'],
            tree.techs['seafaring'],
        ])

    def testRequirementsNotMet(self):
        tree = TechnologyTree()
        tree.research('alphabet')
        tree.research('pottery')
        tree.research('map_making')
        with self.assertRaises(ValueError):
            tree.research('construction')

    def testAlreadyResearched(self):
        tree = TechnologyTree()
        tree.research('alphabet')
        tree.research('pottery')
        tree.research('map_making')
        with self.assertRaises(ValueError):
            tree.research('alphabet')

    def testDoesNotExist(self):
        tree = TechnologyTree()
        tree.research('alphabet')
        tree.research('pottery')
        tree.research('map_making')
        with self.assertRaises(ValueError):
            tree.research('rocket_science')

class TestPacketFactory(unittest.TestCase):
    def testHello(self):
        bytestream = b'\x00\x00hello\x00'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['greeting'], 'hello')

    def testHelloReply(self):
        bytestream = b'\x00\x01helloreply\x00'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['greeting'], 'helloreply')

    def testMap(self):
        bytestream = b'\x00\x02\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertListEqual(packet.content['map'].tolist(), [1, 2, 3])

    def testUnitInfo(self):
        bytestream = b'\x00\x03\x00\x00\x00\x01owner\x00nationality\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['unit_id'], 1)
        self.assertEqual(packet.content['owner'], 'owner')
        self.assertEqual(packet.content['nationality'], 'nationality')
        self.assertEqual(packet.content['coordx'], 2)
        self.assertEqual(packet.content['coordy'], 3)
        self.assertEqual(packet.content['upkeep'], 4)

    def testPlayerInfo(self):
        bytestream = b'\x00\x04\x00\x00\x00\x01playername\x00username\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['playerno'], 1)
        self.assertEqual(packet.content['name'], 'playername')
        self.assertEqual(packet.content['username'], 'username')
        self.assertEqual(packet.content['score'], 2)
        self.assertEqual(packet.content['turns_alive'], 3)
        self.assertEqual(packet.content['is_alive'], 4)
        self.assertEqual(packet.content['gold'], 5)
        self.assertEqual(packet.content['percent_tax'], 6)
        self.assertEqual(packet.content['science'], 7)
        self.assertEqual(packet.content['luxury'], 8)

    def testCityInfo(self):
        bytestream = b'\x00\x05\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\x09\x00\x00\x00\x0aimprovements\x00'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['id'], 1)
        self.assertEqual(packet.content['coordx'], 2)
        self.assertEqual(packet.content['coordy'], 3)
        self.assertEqual(packet.content['owner'], 4)
        self.assertEqual(packet.content['size'], 5)
        self.assertEqual(packet.content['radius'], 6)
        self.assertEqual(packet.content['food_stock'], 7)
        self.assertEqual(packet.content['shield_stock'], 8)
        self.assertEqual(packet.content['production_kind'], 9)
        self.assertEqual(packet.content['production_value'], 10)
        self.assertEqual(packet.content['improvements'], 'improvements')

    def testAction(self):
        bytestream = b'\x00\x06action\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['action'], 'action')
        self.assertEqual(packet.content['ACTION_ID'], 1)
        self.assertEqual(packet.content['actor_id'], 2)
        self.assertEqual(packet.content['target_id'], 3)

    def testActionReply(self):
        bytestream = b'\x00\x07actionreply\x00'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['action'], 'actionreply')

    def testTurnBegin(self):
        bytestream = b'\x00\x08\x00\x00\x00\x01'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['turn_begin'], 1)

    def testTurnEnd(self):
        bytestream = b'\x00\x09turn_end\x00'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['turn_end'], 'turn_end')

    def testCompletedStateTransfer(self):
        bytestream = b'\x00\x0adone\x00'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['done'], 'done')

    def testResearchInfo(self):
        bytestream = b'\x00\x0b\x00\x00\x00\x01\x00\x00\x00\x02researching\x00\x00\x00\x00\x03\x00\x00\x00\x04'
        packet = PacketFactory(bytestream).make_packet()
        encoded = packet.encode()[2:]
        self.assertEqual(encoded, bytestream)
        self.assertEqual(packet.content['id'], 1)
        self.assertEqual(packet.content['techs_researched'], 2)
        self.assertEqual(packet.content['researching'], 'researching')
        self.assertEqual(packet.content['researching_cost'], 3)
        self.assertEqual(packet.content['bulbs_researched'], 4)

if __name__ == '__main__':
    unittest.main()
