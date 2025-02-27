"""Client-end for the ASR demo."""
import os
import struct
import socket
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "fn",
    type=str,
    help="Filename of wav file to transcribe.")
parser.add_argument(
    "--host_ip",
    default="localhost",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
args = parser.parse_args()



def main():

    fn = os.path.abspath(args.fn)
    assert fn and os.path.isfile(fn)

    print('Sending:', fn)

    # Connect to server and send data
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host_ip, args.host_port))
    # sent = ''.join(data_list)
    sent = fn
    sock.sendall(struct.pack('>i', len(sent)) + sent)
    print('Speech[length=%d] Sent.' % len(sent))
    # Receive data from the server and shut down
    received = sock.recv(1024)
    print "Recognition Results: {}".format(received)
    sock.close()

if __name__ == "__main__":
    main()
