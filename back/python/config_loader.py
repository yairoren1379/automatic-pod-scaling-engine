"""Loads the configuration once, for the offline scripts (training, evaluation, experiments).

The API server does not use this module: it keeps a live connection to ZooKeeper and watches
for changes (see engine/zk_store.py).
"""
import json
import logging
import os
import sys

from kazoo.client import KazooClient

from setup_config import DEFAULT_CONFIG, ZK_HOSTS, CONFIG_PATH


def load_zk_config():
    if os.environ.get("ZK_DISABLED") == "1":
        return DEFAULT_CONFIG

    zk = KazooClient(hosts=ZK_HOSTS)
    # Kazoo logs every failed connection retry; we print our own single warning instead.
    kazoo_logger = logging.getLogger("kazoo")
    kazoo_logger.setLevel(logging.CRITICAL)
    try:
        zk.start(timeout=5)
    except Exception as e:
        zk.stop()
        # Offline mode: training in the mock environment does not need a running cluster,
        # so fall back to the defaults instead of refusing to start.
        print(f"WARNING: ZooKeeper at {ZK_HOSTS} is not reachable ({e}).")
        print("WARNING: Using the default configuration from setup_config.py.")
        return DEFAULT_CONFIG
    finally:
        kazoo_logger.setLevel(logging.NOTSET)

    try:
        if not zk.exists(CONFIG_PATH):
            print(f"Error: Path {CONFIG_PATH} does not exist in Zookeeper. Run setup_config.py first.")
            sys.exit(1)

        data, stat = zk.get(CONFIG_PATH)
        config = json.loads(data.decode('utf-8'))
    finally:
        zk.stop()

    missing = [section for section in DEFAULT_CONFIG if section not in config]
    if missing:
        print(f"Error: the config in ZooKeeper is missing {missing}. Run setup_config.py to update it.")
        sys.exit(1)

    return config

APP_CONFIG = load_zk_config()
