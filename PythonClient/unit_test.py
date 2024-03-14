from packets import *

def test(bytestream):
    packet = PacketFactory(bytestream).make_packet()
    for field in packet.field_names:
        print(f'{field}: {packet.content[field]}')
    encoded = packet.encode()[2:] # remove the packet length field before comparison
    if encoded == bytestream:
        print('encoded matches bytestream')
        print(f'\tgot: {encoded}')
    else:
        print('encoded does not match bytestream')
        print(f'\texpected: {bytestream}')
        print(f'\t     got: {encoded}')
    print()

if __name__ == '__main__':
    # hello: 'hello'
    test(b'\x00\x00hello\x00')

    # hello reply: 'helloreply'
    test(b'\x00\x01helloreply\x00')

    # map: [1 2 3]
    test(b'\x00\x02\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03')

    # unit info: id = 10
    test(b'\x00\x03\x00\x00\x00\x0a')

    # civ info: nation_tag = 7
    test(b'\x00\x04\x00\x00\x00\x07')

    # city info: city_name = 'city', pop = 12, owned_by = 'me'
    test(b'\x00\x05city\x00\x00\x00\x00\x0cme\x00')

    # action: action = 'do smth', action_specifiers = 'magestically and philanthropically'
    test(b'\x00\x06do smth\x00magestically and philanthropically\x00')

    # action reply: action = 'pray'
    test(b'\x00\x07pray\x00')

    # turn begin: 2
    test(b'\x00\x08\x00\x00\x00\x02')

    # turn end: '9'
    test(b'\x00\x099\x00')

    # completed state transfer: done = 'yea its done'
    test(b'\x00\x0ayea its done\x00')
