from openreward.environments import Server

from discox import DiscoX

if __name__ == "__main__":
    server = Server([DiscoX])
    server.run()
