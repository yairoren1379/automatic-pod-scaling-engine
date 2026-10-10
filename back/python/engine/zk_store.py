"""Live access to the ZooKeeper ZNodes, with watches.

When a ZNode changes (from the dashboard, from zkCli, from another service), ZooKeeper notifies
every watcher and the new value applies immediately - no restart.

If ZooKeeper is not reachable, the store keeps the values in memory instead, so the API server
and the dashboard still work locally (for example with simulate_controller.py).
"""
import json
import logging
import threading

from kazoo.client import KazooClient

log = logging.getLogger(__name__)


class ZkStore:

    def __init__(self, hosts: str, defaults: dict, timeout: float = 5.0, enabled: bool = True):
        """defaults: {path: value} used when a ZNode does not exist or ZooKeeper is down.
        enabled=False skips ZooKeeper entirely (tests, quick local runs)."""
        self.defaults = defaults
        self.memory = {path: json.loads(json.dumps(value)) for path, value in defaults.items()}
        self.callbacks = {}   # path -> callbacks, used only in memory mode
        self.lock = threading.Lock()
        self.zk = KazooClient(hosts=hosts)
        logging.getLogger("kazoo").setLevel(logging.CRITICAL)
        if not enabled:
            self.connected = False
            return
        try:
            self.zk.start(timeout=timeout)
            self.connected = True
            for path, value in defaults.items():
                if not self.zk.exists(path):
                    self.zk.create(path, json.dumps(value, indent=2).encode(), makepath=True)
        except Exception as e:
            self.zk.stop()
            self.connected = False
            log.warning("ZooKeeper at %s is not reachable (%s): keeping configuration in memory", hosts, e)

    def get(self, path: str) -> dict:
        if self.connected:
            data, _ = self.zk.get(path)
            return json.loads(data.decode())
        with self.lock:
            return json.loads(json.dumps(self.memory[path]))

    def set(self, path: str, value: dict):
        if self.connected:
            self.zk.set(path, json.dumps(value, indent=2).encode())
        else:
            with self.lock:
                self.memory[path] = json.loads(json.dumps(value))
            self._notify(path, value)

    def watch(self, path: str, callback):
        """Calls callback(value) now and after every change of the ZNode."""
        if self.connected:
            @self.zk.DataWatch(path)
            def on_change(data, stat):
                if data is not None:
                    callback(json.loads(data.decode()))
        else:
            self.callbacks.setdefault(path, []).append(callback)
            callback(self.get(path))

    def _notify(self, path, value):
        for callback in self.callbacks.get(path, []):
            callback(value)

    def close(self):
        if self.connected:
            self.zk.stop()
