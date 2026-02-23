from kazoo.client import KazooClient
import json
import sys

def load_zk_config():
    try:
        zk = KazooClient(hosts='127.0.0.1:2181')
        zk.start(timeout=5)
        path = "/autoscaler/config"
        if not zk.exists(path):
            print(f"Error: Path {path} does not exist in Zookeeper.")
            sys.exit(1)
            
        data, stat = zk.get(path)
        zk.stop()
        return json.loads(data.decode('utf-8'))
    except Exception as e:
        print(f"Failed to load config from ZK: {e}")
        sys.exit(1)

APP_CONFIG = load_zk_config()