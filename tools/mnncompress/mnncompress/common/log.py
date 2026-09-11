"""Compatibility API for the retired compression telemetry logger.

Cloud telemetry is disabled. Importing this module does not fetch credentials,
install packages, or collect machine information.
"""


class MNNLogger(object):
    def put_log(self, log_dict, topic=None):
        """Return False because no log is uploaded."""
        return False

    def on_done(self, framework, model_guid, detail):
        if framework not in ["tensorflow", "pytorch"]:
            raise ValueError("framework should be tensorflow or pytorch")
        log_dict = {}
        log_dict["framework"] = framework
        log_dict["model_guid"] = model_guid
        log_dict["detail"] = detail
        return self.put_log(log_dict)


mnn_logger = MNNLogger()
