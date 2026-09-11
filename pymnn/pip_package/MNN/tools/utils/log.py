"""Compatibility API for the retired converter and quantizer telemetry logger.

Cloud telemetry is disabled. Importing this module does not fetch credentials,
install packages, or collect machine information.
"""


class MNNLogger(object):
    def put_log(self, log_dict, topic):
        """Return False because no log is uploaded."""
        return False


mnn_logger = MNNLogger()
